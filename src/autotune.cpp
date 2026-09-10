/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 */

// This whole file sweeps kind/blk/unroll/rpt configurations across the
// explicit LDG/TMA kernel families by launching them directly against the
// SparseMatrix's sliced-ELL device arrays. Those arrays are typed pointers
// only under EXPLICIT_KERNELS; under the default build they are width-agnostic
// void* dispatched through IndexMode (see SparseMatrix.hpp), which none of the
// launchers below understand. There is also nothing to tune there: the
// default build routes SpMV/SymGS through cuSPARSE/NVPL, not these kernels.
#if defined(USE_CUDA) && defined(EXPLICIT_KERNELS)

#include "CudaKernels.hpp"
#include "Geometry.hpp"
#include "SparseMatrix.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace
{
struct Cand
{
    int kind;
    int blk;
    int unroll;
    int rpt;
    // MV LDG_V2 only: number of row partitions across blockIdx.x (1 = the
    // original flat 1-D grid). Every other family ignores it and leaves it at 1.
    int parts = 1;
};
constexpr float kInf = 1e30f;

// The winner line names one config per family, which answers "what should this
// level use" but not "how much did the choice matter". Without the second
// answer a policy that wins by 0.05% and one that wins by 4% look identical in
// the log, and a run-to-run flip between two near-tied configs reads as a
// finding. Both were misread that way before this existed.
//
// Every candidate is already timed, so recording the best under each policy and
// each partition count costs nothing but the bookkeeping below.
struct V3Split
{
    static constexpr int kParts = 6; // PART = 1, 2, 4, 8, 16, 32
    float pol[2];
    Cand polc[2];
    float part[kParts];

    // Every timed config, kept so that a launch shape can be looked up after
    // the sweep. The winning family is not known until the loop ends, so the
    // cross-family comparison below cannot be accumulated on the fly.
    struct Timed
    {
        Cand c;
        float t;
    };
    std::vector<Timed> all;

    void reset()
    {
        for (int i = 0; i < 2; ++i)
        {
            pol[i] = kInf;
            polc[i] = Cand{};
        }
        for (int i = 0; i < kParts; ++i)
            part[i] = kInf;
        all.clear();
    }

    void add(const Cand& c, float t)
    {
        if (t < 0.0f)
            return;
        all.push_back(Timed{c, t});
        if (c.kind != SV_KIND_LDGV3)
            return;
        const int p = SvRptCached(c.rpt) ? 1 : 0;
        if (t < pol[p])
        {
            pol[p] = t;
            polc[p] = c;
        }
        int idx = 0;
        for (int v = SvRptPart(c.rpt); v > 1; v >>= 1)
            ++idx;
        if (idx >= 0 && idx < kParts && t < part[idx])
            part[idx] = t;
    }

    // Always names the winner and quotes a positive margin. A signed delta
    // would need the reader to remember which policy the sign is relative to,
    // and the two lines sit next to a winner column that already uses %+.1f
    // for something else.
    //
    // A third line gives LDG3 against every other family, signed as
    // (LDG3 - family) / family, so positive means LDG3 is slower by that much
    // and negative means it is faster. The winner line already carries each
    // family's time, but reading a gap off it means dividing two four-decimal
    // numbers by eye, per family, per slot, and the answer to "can LDG3 replace
    // this one" is the gap rather than the times.
    //
    // Comparing only against the slot winner would print nothing useful
    // wherever LDG3 already wins, which is half the slots. Under
    // HPCG_FORCE_KIND=4 nothing else is timed and the line is omitted.
    void print(bool with_part, const float* best, const char* const* kname) const
    {
        if (pol[0] >= kInf && pol[1] >= kInf)
            return;
        char sbuf[40] = "n/a", cbuf[40] = "n/a";
        if (pol[0] < kInf)
            snprintf(sbuf, sizeof sbuf, "%d/%d/%d%s=%.4f", polc[0].blk, polc[0].unroll, SvRptWidth(polc[0].rpt),
                SvRptTag(polc[0].rpt), pol[0]);
        if (pol[1] < kInf)
            snprintf(cbuf, sizeof cbuf, "%d/%d/%d%s=%.4f", polc[1].blk, polc[1].unroll, SvRptWidth(polc[1].rpt),
                SvRptTag(polc[1].rpt), pol[1]);
        printf("       LDG3 policy: stream %-20s cached %-20s", sbuf, cbuf);
        if (pol[0] < kInf && pol[1] < kInf)
        {
            const bool cached_wins = pol[1] < pol[0];
            const float lo = cached_wins ? pol[1] : pol[0];
            const float hi = cached_wins ? pol[0] : pol[1];
            printf(" %s wins by %.2f%%", cached_wins ? "cached" : "stream", 100.0 * (hi - lo) / hi);
        }
        printf("\n");

        const float v3 = pol[0] < pol[1] ? pol[0] : pol[1];
        if (v3 < kInf)
        {
            bool any_rival = false;
            for (int k = 0; k < kNumSvKernelKinds; ++k)
                if (k != SV_KIND_LDGV3 && best[k] < kInf)
                    any_rival = true;
            if (any_rival)
            {
                printf("       LDG3 vs:    ");
                for (int k = 0; k < kNumSvKernelKinds; ++k)
                    if (k != SV_KIND_LDGV3 && best[k] < kInf)
                        printf(" %s %+.2f%%", kname[k], 100.0 * (v3 - best[k]) / best[k]);
                printf("   (+ is LDG3 slower)\n");
            }
        }
        if (!with_part)
            return;
        bool any = false;
        for (int i = 0; i < kParts; ++i)
            if (part[i] < kInf && i != 3)
                any = true;
        if (!any)
            return; // PART was not swept, so the only entry is the default 8
        printf("       LDG3 part:  ");
        for (int i = 0; i < kParts; ++i)
            if (part[i] < kInf)
                printf(" p%-2d %.4f", 1 << i, part[i]);
        printf("\n");
    }

    // Best time for one family at one launch shape, or kInf when that family
    // has nothing feasible there. LDG carries no rows-per-thread, so its rpt
    // stays 0 and it never matches a shape that has one; that prints as n/a,
    // which is the honest answer rather than a silent mismatch.
    float at(int kind, const Cand& shape) const
    {
        float bt = kInf;
        for (const Timed& e : all)
            if (e.c.kind == kind && e.c.blk == shape.blk && e.c.unroll == shape.unroll
                && SvRptWidth(e.c.rpt) == SvRptWidth(shape.rpt) && e.t < bt)
                bt = e.t;
        return bt;
    }

    // When another family takes a slot, two questions are tangled: is its
    // launch shape better, or is its memory path better? LDG3 and the TMA
    // families read BLKDIM/UNROLL/rows-per-thread the same way, and both
    // advance 2*UNROLL k-steps per iteration, so a shape transfers between them
    // unchanged. Timing each family at both shapes separates the two.
    //
    // What this cannot show: the sweep already minimises over LDG3, so LDG3 at
    // the rival's shape is never faster than LDG3's own pick, and a near-tie
    // there would be arithmetic, not evidence. The informative halves are how
    // much LDG3 loses when forced into the rival's shape, which prices the
    // shape, and how the rival fares at LDG3's shape, which prices the memory
    // path with the geometry held fixed.
    void print_geometry(const Cand& w, float wt, const Cand& v3c, float v3t, const char* const* kname) const
    {
        if (w.kind == SV_KIND_LDGV3 || wt >= kInf || v3t >= kInf)
            return;
        const float v3_there = at(SV_KIND_LDGV3, w);
        const float riv_here = at(w.kind, v3c);
        // LDG is templated on block and unroll only, so it shares no shape with
        // LDG3 and both halves come back empty. That line says nothing.
        if (v3_there >= kInf && riv_here >= kInf)
            return;
        printf("       geometry:    at %d/%d/%d %s %.4f LDG3 ", w.blk, w.unroll, SvRptWidth(w.rpt), kname[w.kind], wt);
        if (v3_there < kInf)
            printf("%.4f (%+.2f%%)", v3_there, 100.0 * (v3_there - wt) / wt);
        else
            printf("n/a");
        printf("  |  at %d/%d/%d LDG3 %.4f %s ", v3c.blk, v3c.unroll, SvRptWidth(v3c.rpt), v3t, kname[w.kind]);
        if (riv_here < kInf)
            printf("%.4f (%+.2f%%)", riv_here, 100.0 * (riv_here - v3t) / v3t);
        else
            printf("n/a");
        printf("\n");
    }

    // Every candidate of one family, worst to best. For a one-off question
    // ("time every LDG3 and every TMA shape at this slot") rather than the
    // steady-state autotune print, which only ever names the winner.
    void dump_all(int kind, const char* label) const
    {
        std::vector<Timed> sub;
        for (const Timed& e : all)
            if (e.c.kind == kind)
                sub.push_back(e);
        std::sort(sub.begin(), sub.end(), [](const Timed& a, const Timed& b) { return a.t > b.t; });
        for (const Timed& e : sub)
            printf("  DUMP %-5s %d/%d/%d%s = %.4f\n", label, e.c.blk, e.c.unroll, SvRptWidth(e.c.rpt),
                SvRptTag(e.c.rpt), e.t);
    }
};
}

void AutotuneSymGS(const SparseMatrix& A_top)
{
    const char* en = std::getenv("HPCG_AUTOTUNE");
    if (!en || std::atoi(en) == 0)
        return;

    const int rank = A_top.geom ? A_top.geom->rank : 0;

    int iters = 20;
    const char* it = std::getenv("HPCG_AUTOTUNE_ITERS");
    if (it && *it)
        iters = std::atoi(it);
    if (iters < 1)
        iters = 1;

    int force_kind = -1;
    const char* fk = std::getenv("HPCG_FORCE_KIND");
    if (fk && *fk)
        force_kind = std::atoi(fk);

    std::vector<Cand> cands;
    int n_reg = 0, n_tma = 0, n_v2 = 0;
    const int reg_blk[] = {64, 128, 256};
    const int reg_un[] = {4, 6, 7, 8, 10, 12, 14, 16};
    for (int b : reg_blk)
        for (int u : reg_un)
        {
            cands.push_back({SV_KIND_LDG, b, u, 0});
            ++n_reg;
        }
    const int tma_blk[] = {32, 64, 128, 256, 512};
    const int tma_un[] = {1, 2, 3, 4, 6, 8};
    const int tma_rpt[] = {1, 2, 4, 8, 16};
    for (int b : tma_blk)
        for (int u : tma_un)
            for (int r : tma_rpt)
            {
                cands.push_back({SV_KIND_TMA, b, u, r});
                ++n_tma;
            }
    const int v2_blk[] = {32, 64, 128, 256};
    const int v2_un[] = {1, 2, 3, 4};
    const int v2_w[] = {1, 2, 4, 8};
    for (int b : v2_blk)
        for (int u : v2_un)
            for (int w : v2_w)
            {
                cands.push_back({SV_KIND_LDGV2, b, u, w});
                ++n_v2;
            }

    int n_t2 = 0;
    const int t2_blk[] = {32, 64, 128, 256};
    const int t2_un[] = {1, 2, 3, 4, 6, 8};
    const int t2_rpt[] = {1, 2, 4};
    for (int b : t2_blk)
        for (int u : t2_un)
            for (int r : t2_rpt)
            {
                cands.push_back({SV_KIND_TMA2D, b, u, r});
                ++n_t2;
            }

    // LDG3 starts from LDG_V2's blk/unroll/W grid and extends it on three axes:
    // cache policy, access width, and unroll depth at small W.
    //
    // LDG3 has two axes LDG_V2 does not, and both describe how W is fetched
    // rather than naming a different kernel, so both live here and
    // HPCG_FORCE_KIND=4 searches all of them.
    //
    // Access width: a negative W asks for the widest load available rather than
    // 128-bit accesses. It only means something at 4 and 8 rows per thread,
    // since 1 and 2 are already 8 and 16 bytes, so the negative entries stop
    // there.
    //
    // Cache policy: +100 asks for ordinary cached loads instead of the streaming
    // __ldcs. This is the axis that decides LDG3 against LDG_V2. LDG3 took its
    // policy from LDG, which streams, and LDG_V2 keeps the line; measurement put
    // LDG_V2 ahead at every level 3 slot and LDG3 ahead at levels 0 and 1, which
    // is what a working set of 46 MiB against 23 GiB predicts. Making the policy
    // a candidate rather than a property is what lets one family cover both.
    const int v3_w[] = {1, 2, 4, 8, -4, -8, 101, 102, 104, 108, -104, -108};
    int n_v3 = 0;
    for (int b : v2_blk)
        for (int u : v2_un)
            for (int w : v3_w)
            {
                cands.push_back({SV_KIND_LDGV3, b, u, w});
                ++n_v3;
            }

    // Depth at small W, LDG3 only. LDG takes level 2 at 128/14 and 256/10 with
    // one row per thread, and LDG_V2's grid stops at 4, so LDG3 inherited a cap
    // that made those configurations unreachable rather than merely unchosen.
    // The unit differs between the two: LDG advances Unroll k-steps per loop
    // iteration where LDG3's A/B pair advances 2*UNROLL, so LDG's 14 and 10 are
    // UNROLL 7 and 5 here, and the list brackets both. Depth costs
    // 2*UNROLL*W*12 bytes of register file, which is why it is offered at W of 1
    // and 2 and nowhere else -- at W of 8 the existing cap of 4 already sits
    // near 190 registers. Both cache policies, no wide, which needs W >= 4.
    const int v3_deep_u1[] = {5, 6, 7, 8, 10, 14};
    const int v3_deep_u2[] = {5, 6, 7, 8};
    for (int b : v2_blk)
    {
        for (int u : v3_deep_u1)
            for (int w : {1, 101})
            {
                cands.push_back({SV_KIND_LDGV3, b, u, w});
                ++n_v3;
            }
        for (int u : v3_deep_u2)
            for (int w : {2, 102})
            {
                cands.push_back({SV_KIND_LDGV3, b, u, w});
                ++n_v3;
            }
    }

    // TMA_EX is templated on block and unroll only -- one row per thread is
    // built into it -- so rpt stays at 1 and the pairs below are exactly the
    // ones the dispatcher instantiates. Anything else would only be a launch
    // that returns false.
    const int tx_sv[][2] = {{64, 4}, {64, 6}, {64, 7}, {64, 8}, {64, 10}, {64, 12}, {128, 4}, {128, 6}, {128, 8},
        {256, 4}, {256, 6}, {256, 7}};
    int n_tx = 0;
    for (const auto& p : tx_sv)
    {
        cands.push_back({SV_KIND_TMAEX, p[0], p[1], 1});
        ++n_tx;
    }

    std::vector<Cand> mvcands;
    int m_reg = 0, m_tma = 0, m_v2 = 0;
    const int mreg_un[] = {1, 4, 6, 7, 8, 10, 12, 14, 16};
    for (int b : reg_blk)
        for (int u : mreg_un)
        {
            mvcands.push_back({SV_KIND_LDG, b, u, 0});
            ++m_reg;
        }
    const int mtma_blk[] = {32, 64, 128, 256};
    const int mtma_un[] = {1, 2, 3, 4, 6, 8};
    const int mtma_rpt[] = {1, 2, 4, 8};
    for (int b : mtma_blk)
        for (int u : mtma_un)
            for (int r : mtma_rpt)
            {
                mvcands.push_back({SV_KIND_TMA, b, u, r});
                ++m_tma;
            }
    // MV LDG_V2 additionally sweeps the row-partition count: parts=1 is the
    // original flat 1-D grid, parts>1 reproduces and generalises LDG3's
    // hard-coded 8-way split. Infeasible combinations (m not divisible, or
    // gridDim.y over 65535) are rejected at launch and time as -1.
    const int v2_parts[] = {1, 2, 4, 8, 16};
    for (int b : v2_blk)
        for (int u : v2_un)
            for (int w : v2_w)
                for (int pt : v2_parts)
                {
                    mvcands.push_back({SV_KIND_LDGV2, b, u, w, pt});
                    ++m_v2;
                }
    int m_t2 = 0;
    for (int b : t2_blk)
        for (int u : t2_un)
            for (int r : t2_rpt)
            {
                mvcands.push_back({SV_KIND_TMA2D, b, u, r});
                ++m_t2;
            }
    int m_v3 = 0;
    for (int b : v2_blk)
        for (int u : v2_un)
            for (int w : v3_w)
            {
                mvcands.push_back({SV_KIND_LDGV3, b, u, w});
                ++m_v3;
            }
    for (int b : v2_blk)
    {
        for (int u : v3_deep_u1)
            for (int w : {1, 101})
            {
                mvcands.push_back({SV_KIND_LDGV3, b, u, w});
                ++m_v3;
            }
        for (int u : v3_deep_u2)
            for (int w : {2, 102})
            {
                mvcands.push_back({SV_KIND_LDGV3, b, u, w});
                ++m_v3;
            }
    }
    // The MV row-partition count, opt-in because it multiplies the LDG3 half of
    // the sweep rather than adding to it. PART is a launch shape and costs no
    // extra instantiations, but every value re-times the whole LDG3 grid at
    // every level and in all three directions, so leaving it on by default
    // would triple a sweep whose current cost is already the LDG3 family.
    //
    // 8 is the historical value and is already present as the unencoded form,
    // so only the others are added. 1 is LDG_V2's plain 1D walk, which is what
    // the coarse levels appear to want; 16 and 32 test the other direction at
    // level 0, where nothing has ever checked whether 8 was the right choice.
    if (const char* sp = std::getenv("HPCG_SWEEP_PART"))
        if (*sp && std::atoi(sp) != 0)
        {
            // The LDG3 candidates are the last m_v3 entries right now; TMA_EX is
            // appended after this. The base range is snapshotted by index before
            // any push_back, since growing the vector invalidates references into
            // it and would also walk the newly added entries.
            const int base_v3 = m_v3;
            const size_t v3_begin = mvcands.size() - (size_t) base_v3;
            for (int f : {1, 2, 3, 5, 6}) // PART = 1, 2, 4, 16, 32
                for (int i = 0; i < base_v3; ++i)
                {
                    const Cand c = mvcands[v3_begin + (size_t) i];
                    const int mag = (c.rpt < 0 ? -c.rpt : c.rpt) + 1000 * f;
                    mvcands.push_back({SV_KIND_LDGV3, c.blk, c.unroll, c.rpt < 0 ? -mag : mag});
                }
            m_v3 += 5 * base_v3;
        }
    const int tx_mv[][2] = {{64, 1}, {64, 4}, {64, 6}, {64, 7}, {64, 8}, {128, 1}, {128, 4}, {128, 6}, {128, 7},
        {128, 8}, {256, 1}, {256, 4}, {256, 6}, {256, 7}};
    int m_tx = 0;
    for (const auto& p : tx_mv)
    {
        mvcands.push_back({SV_KIND_TMAEX, p[0], p[1], 1});
        ++m_tx;
    }

    // Per-operator overrides. A family need not serve both operators -- an MV
    // kernel with no triangular-solve counterpart is a legitimate thing to
    // evaluate -- and forcing it globally would leave SV with an empty candidate
    // list, whereupon every level reports "no feasible config" and quietly keeps
    // the heuristic pick. That reads as a single-family run but is a blend.
    int force_mv = force_kind, force_sv = force_kind;
    if (const char* e = std::getenv("HPCG_FORCE_KIND_MV"))
        if (*e)
            force_mv = std::atoi(e);
    if (const char* e = std::getenv("HPCG_FORCE_KIND_SV"))
        if (*e)
            force_sv = std::atoi(e);

    // TMA_EX stays out of the unrestricted search unless asked for. Adding a
    // sixth family to the default sweep would change what the free-sweep arm
    // measures, so every number already recorded for it would belong to a
    // different experiment while still carrying the same name. Forcing the
    // family obviously still searches it.
    const char* sw = std::getenv("HPCG_SWEEP_TMAEX");
    const bool sweep_tx = sw && *sw && std::atoi(sw) != 0;
    /*
      HPCG_TUNE_DIRS: 0 off, 1 measure and report only, 2 measure and install.

      Kept apart deliberately. Every directional run recorded so far was taken
      under =1, whose contract is that the selection is untouched; promoting it
      silently would leave those runs incomparable with future ones while
      looking identical in the log. =2 is the opt-in that claims the 5%.
    */
    const char* sd = std::getenv("HPCG_TUNE_DIRS");
    const int tune_dirs = sd && *sd ? std::atoi(sd) : 0;
    const bool sweep_dirs = tune_dirs != 0;
    const bool select_dirs = tune_dirs >= 2;
    auto drop_tmaex = [](std::vector<Cand>& v, int forced) {
        if (forced == SV_KIND_TMAEX)
            return;
        v.erase(std::remove_if(v.begin(), v.end(), [](const Cand& c) { return c.kind == SV_KIND_TMAEX; }), v.end());
    };
    if (!sweep_tx)
    {
        drop_tmaex(cands, force_sv);
        drop_tmaex(mvcands, force_mv);
        if (force_sv != SV_KIND_TMAEX)
            n_tx = 0;
        if (force_mv != SV_KIND_TMAEX)
            m_tx = 0;
    }

    // TMA2D has never won a slot in any sweep to date (0 of 32) and at large
    // local problem sizes cuTensorMapEncodeTiled fails with a misaligned
    // address; that failure corrupts the CUDA context for the rest of the
    // process (every later CUDA call, including cudaMemGetInfo, then fails
    // too), taking the whole run down rather than just losing TMA2D's slot.
    // Off by default; HPCG_SWEEP_TMA2D=1 re-enables it for investigation.
    const char* s2 = std::getenv("HPCG_SWEEP_TMA2D");
    const bool sweep_t2 = s2 && *s2 && std::atoi(s2) != 0;
    auto drop_tma2d = [](std::vector<Cand>& v, int forced) {
        if (forced == SV_KIND_TMA2D)
            return;
        v.erase(std::remove_if(v.begin(), v.end(), [](const Cand& c) { return c.kind == SV_KIND_TMA2D; }), v.end());
    };
    if (!sweep_t2)
    {
        drop_tma2d(cands, force_sv);
        drop_tma2d(mvcands, force_mv);
        if (force_sv != SV_KIND_TMA2D)
            n_t2 = 0;
        if (force_mv != SV_KIND_TMA2D)
            m_t2 = 0;
    }

    auto keep = [](std::vector<Cand>& v, int kind) {
        if (kind < 0)
            return;
        std::vector<Cand> f;
        for (const Cand& c : v)
            if (c.kind == kind)
                f.push_back(c);
        v.swap(f);
    };
    keep(cands, force_sv);
    keep(mvcands, force_mv);
    if (rank == 0)
    {
        if (force_sv == force_mv && force_sv >= 0)
            printf("[autotune] HPCG_FORCE_KIND=%d: restricting the search to that kernel family\n", force_sv);
        else
        {
            if (force_sv >= 0)
                printf("[autotune] HPCG_FORCE_KIND_SV=%d: SymGS restricted to that family\n", force_sv);
            if (force_mv >= 0)
                printf("[autotune] HPCG_FORCE_KIND_MV=%d: SpMV restricted to that family\n", force_mv);
        }
        // An empty list is not the same as a family that loses; say so here
        // rather than leaving four "no feasible config" lines to be interpreted.
        if (cands.empty())
            printf("[autotune] WARNING: no SymGS candidates after filtering; SymGS keeps its heuristic pick\n");
        if (mvcands.empty())
            printf("[autotune] WARNING: no SpMV candidates after filtering; SpMV keeps its heuristic pick\n");
    }

    const size_t n = (size_t) A_top.localNumberOfColumns;
    double* rv = nullptr;
    double* xv = nullptr;
    if (cudaMalloc((void**) &rv, n * sizeof(double)) != cudaSuccess
        || cudaMalloc((void**) &xv, n * sizeof(double)) != cudaSuccess)
    {
        if (rank == 0)
            fprintf(stderr, "[autotune] scratch alloc failed; skipping autotune\n");
        return;
    }
    cudaMemset(rv, 0, n * sizeof(double));
    cudaMemset(xv, 0, n * sizeof(double));

    if (rank == 0)
        printf("\n===== HPCG SymGS (SV) autotune: %d LDG + %d TMA + %d LDG_V2 + %d TMA2D + %d LDG3 + %d TMA_EX "
               "configs/level, iters=%d =====\n",
            n_reg, n_tma, n_v2, n_t2, n_v3, n_tx, iters);

    for (const SparseMatrix* m = &A_top; m != nullptr; m = m->Ac)
    {
        float best[kNumSvKernelKinds];
        Cand bestc[kNumSvKernelKinds];
        for (int k = 0; k < kNumSvKernelKinds; ++k)
        {
            best[k] = kInf;
            bestc[k] = Cand{k, 0, 0, 0};
        }
        V3Split split;
        split.reset();
        for (const Cand& c : cands)
        {
            const float t = TimeSvConfig(*m, rv, xv, c.kind, c.blk, c.unroll, c.rpt, iters);
            if (t < 0.0f)
                continue;
            split.add(c, t);
            if (t < best[c.kind])
            {
                best[c.kind] = t;
                bestc[c.kind] = c;
            }
        }

        int win = 0;
        for (int k = 1; k < kNumSvKernelKinds; ++k)
            if (best[k] < best[win])
                win = k;
        if (best[win] >= kInf)
        {
            if (rank == 0)
                printf("  level %d: no feasible config (skipped)\n", m->level);
            continue;
        }
        SetSvChoice(m->level, bestc[win].kind, bestc[win].blk, bestc[win].unroll, bestc[win].rpt);

        if (rank == 0)
        {
            const char* kname[kNumSvKernelKinds] = {"LDG", "TMA", "LDG_V2", "TMA2D", "LDG3", "TMA_EX"};
            char buf[kNumSvKernelKinds][40], winbuf[40];
            for (int k = 0; k < kNumSvKernelKinds; ++k)
            {
                if (best[k] >= kInf)
                    snprintf(buf[k], sizeof buf[k], "n/a");
                else if (k == SV_KIND_LDG)
                    snprintf(buf[k], sizeof buf[k], "%d/%d=%.4f", bestc[k].blk, bestc[k].unroll, best[k]);
                else
                    snprintf(buf[k], sizeof buf[k], "%d/%d/%d%s=%.4f", bestc[k].blk, bestc[k].unroll,
                        SvRptWidth(bestc[k].rpt), SvRptTag(bestc[k].rpt), best[k]);
            }
            snprintf(winbuf, sizeof winbuf, "%s %d/%d/%d%s", kname[win], bestc[win].blk, bestc[win].unroll,
                SvRptWidth(bestc[win].rpt), SvRptTag(bestc[win].rpt));
            float second = kInf;
            for (int k = 0; k < kNumSvKernelKinds; ++k)
                if (k != win && best[k] < second)
                    second = best[k];
            const double margin = (second < kInf) ? 100.0 * (second - best[win]) / second : 0.0;

            const double entry_bytes = (double) (sizeof(local_int_t) + sizeof(double));
            const double bytes = (double) (m->localNumberOfNonzeros - m->localNumberOfRows) * entry_bytes;
            const double bw = bytes / best[win] / 1.0e6;
            char line[kNumSvKernelKinds * 48];
            int off = 0;
            for (int k = 0; k < kNumSvKernelKinds; ++k)
                off += snprintf(line + off, sizeof line - off, "%s %-16s ", kname[k], buf[k]);
            printf("  L%d rows=%-9d | %s| WIN %-13s (%+.1f%%) BW=%.0f\n", m->level,
                (int) m->localNumberOfRows, line, winbuf, margin, bw);
            split.print(false, best, kname); // SpSV has no row-partition split
            split.print_geometry(bestc[win], best[win], bestc[SV_KIND_LDGV3], best[SV_KIND_LDGV3], kname);
            const char* dl0 = std::getenv("HPCG_DUMP_SV_L0");
            if (dl0 && std::atoi(dl0) != 0 && m->level == 0)
            {
                // HPCG_DUMP_SV_L0_DIR selects which sweep is timed: unset or
                // "both" reuses the combined-sweep times already collected
                // above (matches every prior dump); "fwd"/"bwd" re-times each
                // LDG3/TMA candidate in isolation, since forward runs over L
                // and backward over U -- different submatrices that the
                // combined number cannot tell apart.
                const char* dirEnv = std::getenv("HPCG_DUMP_SV_L0_DIR");
                const int sweep = (dirEnv && !std::strcmp(dirEnv, "fwd")) ? 1
                    : (dirEnv && !std::strcmp(dirEnv, "bwd"))             ? 2
                                                                           : 0;
                if (sweep == 0)
                {
                    split.dump_all(SV_KIND_LDGV3, "LDG3");
                    split.dump_all(SV_KIND_TMA, "TMA");
                }
                else
                {
                    std::vector<V3Split::Timed> iso;
                    for (const Cand& c : cands)
                    {
                        if (c.kind != SV_KIND_LDGV3 && c.kind != SV_KIND_TMA)
                            continue;
                        const float t = TimeSvConfig(*m, rv, xv, c.kind, c.blk, c.unroll, c.rpt, iters, sweep);
                        if (t >= 0.0f)
                            iso.push_back(V3Split::Timed{c, t});
                    }
                    std::sort(iso.begin(), iso.end(),
                        [](const V3Split::Timed& a, const V3Split::Timed& b) { return a.t > b.t; });
                    for (const V3Split::Timed& e : iso)
                        printf("  DUMP-%s %-5s %d/%d/%d%s = %.4f\n", sweep == 1 ? "FWD" : "BWD",
                            e.c.kind == SV_KIND_LDGV3 ? "LDG3" : "TMA", e.c.blk, e.c.unroll, SvRptWidth(e.c.rpt),
                            SvRptTag(e.c.rpt), e.t);
                }
            }
        }
    }

    if (rank == 0)
        printf("\n===== HPCG SpMV (MV) autotune: %d LDG + %d TMA + %d LDG_V2 + %d TMA2D + %d LDG3 + %d TMA_EX "
               "configs/level, iters=%d =====\n",
            m_reg, m_tma, m_v2, m_t2, m_v3, m_tx, iters);

    for (const SparseMatrix* m = &A_top; m != nullptr; m = m->Ac)
    {
        float best[kNumSvKernelKinds];
        Cand bestc[kNumSvKernelKinds];
        for (int k = 0; k < kNumSvKernelKinds; ++k)
        {
            best[k] = kInf;
            bestc[k] = Cand{k, 0, 0, 0};
        }
        V3Split split;
        split.reset();
        for (const Cand& c : mvcands)
        {
            const float t = TimeMvConfig(*m, xv, rv, c.kind, c.blk, c.unroll, c.rpt, c.parts, iters);
            if (t < 0.0f)
                continue;
            split.add(c, t);
            if (t < best[c.kind])
            {
                best[c.kind] = t;
                bestc[c.kind] = c;
            }
        }

        int win = 0;
        for (int k = 1; k < kNumSvKernelKinds; ++k)
            if (best[k] < best[win])
                win = k;
        if (best[win] >= kInf)
        {
            if (rank == 0)
                printf("  level %d: no feasible MV config (skipped)\n", m->level);
            continue;
        }
        SetMvChoice(
            m->level, bestc[win].kind, bestc[win].blk, bestc[win].unroll, bestc[win].rpt, bestc[win].parts);

        if (rank == 0)
        {
            const char* kname[kNumSvKernelKinds] = {"LDG", "TMA", "LDG_V2", "TMA2D", "LDG3", "TMA_EX"};
            char buf[kNumSvKernelKinds][40], winbuf[40];
            for (int k = 0; k < kNumSvKernelKinds; ++k)
            {
                if (best[k] >= kInf)
                    snprintf(buf[k], sizeof buf[k], "n/a");
                else if (k == SV_KIND_LDG)
                    snprintf(buf[k], sizeof buf[k], "%d/%d=%.4f", bestc[k].blk, bestc[k].unroll, best[k]);
                else if (k == SV_KIND_LDGV2)
                    // trailing pN is the winning partition count
                    snprintf(buf[k], sizeof buf[k], "%d/%d/%d p%d=%.4f", bestc[k].blk, bestc[k].unroll, bestc[k].rpt,
                        bestc[k].parts, best[k]);
                else
                    snprintf(buf[k], sizeof buf[k], "%d/%d/%d%s=%.4f", bestc[k].blk, bestc[k].unroll,
                        SvRptWidth(bestc[k].rpt), SvRptTag(bestc[k].rpt), best[k]);
            }
            if (win == SV_KIND_LDGV2)
                snprintf(winbuf, sizeof winbuf, "%s %d/%d/%d p%d", kname[win], bestc[win].blk, bestc[win].unroll,
                    bestc[win].rpt, bestc[win].parts);
            else
                snprintf(winbuf, sizeof winbuf, "%s %d/%d/%d%s", kname[win], bestc[win].blk, bestc[win].unroll,
                    SvRptWidth(bestc[win].rpt), SvRptTag(bestc[win].rpt));
            float second = kInf;
            for (int k = 0; k < kNumSvKernelKinds; ++k)
                if (k != win && best[k] < second)
                    second = best[k];
            const double margin = (second < kInf) ? 100.0 * (second - best[win]) / second : 0.0;

            const double entry_bytes = (double) (sizeof(local_int_t) + sizeof(double));
            const double bytes = (double) m->localNumberOfNonzeros * entry_bytes;
            const double bw = bytes / best[win] / 1.0e6;
            char line[kNumSvKernelKinds * 48];
            int off = 0;
            for (int k = 0; k < kNumSvKernelKinds; ++k)
                off += snprintf(line + off, sizeof line - off, "%s %-16s ", kname[k], buf[k]);
            printf("  L%d rows=%-9d | %s| WIN %-13s (%+.1f%%) BW=%.0f\n", m->level,
                (int) m->localNumberOfRows, line, winbuf, margin, bw);
            split.print(true, best, kname);
            split.print_geometry(bestc[win], best[win], bestc[SV_KIND_LDGV3], best[SV_KIND_LDGV3], kname);
        }
    }

    // Off by default. The MV winner above was chosen by timing the full matrix
    // A, but that one choice is then used for the L and U multiplies inside
    // SymGS too -- half the nonzeros each, and L accumulates where A and U
    // overwrite. Those two run in the MG hot path and were never measured.
    //
    // They also do not want A's shape. On a Rubin uGPU at 512x512x288, A takes
    // 64/7/2p16 at L0 where L takes 128/4/4wp8 and U 128/4/4wcp8; inheriting
    // costs 6.3% and 3.8% there, and 5.0% over all eight slots.
    //
    // Under =1 this only prints what would have been picked, which is what the
    // recorded directional runs mean. Under =2 it installs them as well.
    if (sweep_dirs)
    {
        for (int dir = 1; dir <= 2; ++dir)
        {
            const char* tag = (dir == 1) ? "MV-L" : "MV-U";
            if (rank == 0)
                printf("\n===== MV on the %s submatrix, %s =====\n",
                    (dir == 1) ? "L (SymGS forward, beta=1, accumulates)" : "U (SymGS backward, beta=0, overwrites)",
                    select_dirs ? "SELECTING: winners installed for this triangle" : "diagnostic only, selection unchanged");

            for (const SparseMatrix* m = &A_top; m != nullptr; m = m->Ac)
            {
                float best[kNumSvKernelKinds];
                Cand bestc[kNumSvKernelKinds];
                for (int k = 0; k < kNumSvKernelKinds; ++k)
                {
                    best[k] = kInf;
                    bestc[k] = Cand{k, 0, 0, 0};
                }
                V3Split split;
                split.reset();
                for (const Cand& c : mvcands)
                {
                    const float t = TimeMvConfigDir(*m, xv, rv, c.kind, c.blk, c.unroll, c.rpt, c.parts, iters, dir);
                    if (t < 0.0f)
                        continue;
                    split.add(c, t);
                    if (t < best[c.kind])
                    {
                        best[c.kind] = t;
                        bestc[c.kind] = c;
                    }
                }
                int win = 0;
                for (int k = 1; k < kNumSvKernelKinds; ++k)
                    if (best[k] < best[win])
                        win = k;

                // Before the rank guard: the choice has to exist on every rank
                // that launches the kernel, not just the one that prints.
                if (select_dirs && best[win] < kInf)
                    SetMvChoiceDir(m->level, (dir == 1) ? Forward : Backward, bestc[win].kind, bestc[win].blk,
                        bestc[win].unroll, bestc[win].rpt, bestc[win].parts);

                if (rank != 0)
                    continue;

                if (best[win] >= kInf)
                {
                    printf("  %s L%d rows=%-9d | no feasible config\n", tag, m->level, (int) m->localNumberOfRows);
                    continue;
                }

                const char* kname[kNumSvKernelKinds] = {"LDG", "TMA", "LDG_V2", "TMA2D", "LDG3", "TMA_EX"};
                char buf[kNumSvKernelKinds][40], winbuf[40];
                for (int k = 0; k < kNumSvKernelKinds; ++k)
                {
                    if (best[k] >= kInf)
                        snprintf(buf[k], sizeof buf[k], "n/a");
                    else if (k == SV_KIND_LDG)
                        snprintf(buf[k], sizeof buf[k], "%d/%d=%.4f", bestc[k].blk, bestc[k].unroll, best[k]);
                    else
                        snprintf(buf[k], sizeof buf[k], "%d/%d/%d%s=%.4f", bestc[k].blk, bestc[k].unroll,
                            SvRptWidth(bestc[k].rpt), SvRptTag(bestc[k].rpt), best[k]);
                }
                snprintf(winbuf, sizeof winbuf, "%s %d/%d/%d%s", kname[win], bestc[win].blk, bestc[win].unroll,
                    SvRptWidth(bestc[win].rpt), SvRptTag(bestc[win].rpt));
                float second = kInf;
                for (int k = 0; k < kNumSvKernelKinds; ++k)
                    if (k != win && best[k] < second)
                        second = best[k];
                const double margin = (second < kInf) ? 100.0 * (second - best[win]) / second : 0.0;
                char line[kNumSvKernelKinds * 48];
                int off = 0;
                for (int k = 0; k < kNumSvKernelKinds; ++k)
                    off += snprintf(line + off, sizeof line - off, "%s %-16s ", kname[k], buf[k]);
                printf("  %s L%d rows=%-9d | %s| WIN %-13s (%+.1f%%)\n", tag, m->level,
                    (int) m->localNumberOfRows, line, winbuf, margin);
                split.print(true, best, kname);
                split.print_geometry(bestc[win], best[win], bestc[SV_KIND_LDGV3], best[SV_KIND_LDGV3], kname);
            }
        }
    }

    cudaFree(rv);
    cudaFree(xv);
    if (rank == 0)
        printf("=========================================================================\n\n");
}

#elif defined(USE_CUDA)

#include "CudaKernels.hpp"
#include "SparseMatrix.hpp"

// Default (non-EXPLICIT_KERNELS) build: cuSPARSE/NVPL own SpMV/SymGS, so there
// is nothing here to autotune. Keep the symbol defined so main.cpp does not
// need an EXPLICIT_KERNELS guard just to call it.
void AutotuneSymGS(const SparseMatrix&) {}

#endif
