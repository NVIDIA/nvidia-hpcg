/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
  Per-level kernel selection for the explicit Sliced-ELL families.

  HPCG runs a four-level multigrid hierarchy whose levels differ in working set
  by orders of magnitude, and the family and launch shape that win at the finest
  level are not the ones that win at the coarsest. This sweeps every candidate
  configuration at every level of the hierarchy and installs the fastest one per
  level and per operator, through SetMvChoice / SetSvChoice.

  It measures by launching the kernels directly against the matrix's Sliced-ELL
  device arrays, so it only exists under EXPLICIT_KERNELS: without it those
  arrays are width-agnostic void* dispatched through IndexMode, and SpMV/SymGS
  run on cuSPARSE, where there is nothing here to tune. The whole file compiles
  to an empty object in that build, exactly as the family .cu files do.
*/
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
constexpr float kInf = 1e30f;

/*
  The families this build contains, and nothing else.

  SellKernelKind numbers six families so that a family keeps its number as it
  lands, but only five are implemented here: TMA_EX (5) has no kernels in this
  tree. It is therefore absent from every candidate list and from every column of
  every table below -- a slot no family can serve should not print that family's
  name at all -- and MvSellCfg / SvSellCfg refuse its number outright, so it
  cannot be reached even by naming it in a pin.

  Kept in kind order, so that a column's position in a printed row is the
  family's number. TMA2D is 3 and LDG3 is 4, so TMA2D precedes it here; getting
  that pair the wrong way round would attribute each family's timings to the
  other, which reads as a finding rather than as a bug.
*/
constexpr int kNumFam = 5;
constexpr int kFamKind[kNumFam]
    = {SELL_KIND_LDG, SELL_KIND_TMA, SELL_KIND_LDGV2, SELL_KIND_TMA2D, SELL_KIND_LDGV3};
// Spelled as every analysis in this project already spells them; slotlib.py
// anchors its per-slot regexes on exactly these names.
const char* const kFamName[kNumFam] = {"LDG", "TMA", "LDG_V2", "TMA2D", "LDG3"};

int FamIdx(int kind)
{
    for (int i = 0; i < kNumFam; ++i)
        if (kFamKind[i] == kind)
            return i;
    return -1;
}

/*
  The config tag: "blk/unroll" for LDG, which has no rows-per-thread, and
  "blk/unroll/W" for the rest, with "w" for wide, "c" for cached and "pN" for
  the SpMV partition count appended to W.

  The format is not free: hpcg-lab/bin/slotlib.py parses every autotune log in
  this project, and its config pattern is blk/unroll optionally followed by /rpt
  with those suffixes and no spaces inside them. A tag it cannot match does not
  fail loudly -- the family simply looks absent from that slot and some other
  family looks like the winner -- so it is spelled here to match.
*/
void FormatCfg(char* buf, size_t n, const SellConfig& c, bool with_parts)
{
    if (c.kind == SELL_KIND_LDG)
    {
        snprintf(buf, n, "%d/%d", c.blk, c.unroll);
        return;
    }
    char tag[8];
    int t = 0;
    if (c.wide)
        tag[t++] = 'w';
    if (c.cached)
        tag[t++] = 'c';
    tag[t] = '\0';
    // Partitions are an SpMV launch shape; the triangular solves have no
    // counterpart, so naming one there would print a knob that was never read.
    if (with_parts && (c.kind == SELL_KIND_LDGV2 || c.kind == SELL_KIND_LDGV3))
        snprintf(buf, n, "%d/%d/%d%sp%d", c.blk, c.unroll, c.w, tag, c.parts);
    else
        snprintf(buf, n, "%d/%d/%d%s", c.blk, c.unroll, c.w, tag);
}

struct Timed
{
    SellConfig c;
    float t;
};

/*
  The winner line names one config per family, which answers "what should this
  level use" but not "how much did the choice matter". Without the second answer
  a policy that wins by 0.05% and one that wins by 4% look identical in the log,
  and a run-to-run flip between two near-tied configs reads as a finding. Both
  were misread that way before this existed.

  Every candidate is already timed, so recording the best under each cache
  policy and each partition count costs nothing but the bookkeeping below.
*/
struct Ldg3Split
{
    static constexpr int kParts = 6; // parts = 1, 2, 4, 8, 16, 32
    float pol[2];
    SellConfig polc[2];
    float part[kParts];

    // Every timed config, kept so that a launch shape can be looked up after
    // the sweep. The winning family is not known until the loop ends, so the
    // cross-family comparison below cannot be accumulated on the fly.
    std::vector<Timed> all;

    void reset()
    {
        for (int i = 0; i < 2; ++i)
        {
            pol[i] = kInf;
            polc[i] = SellConfig{};
        }
        for (int i = 0; i < kParts; ++i)
            part[i] = kInf;
        all.clear();
    }

    void add(const SellConfig& c, float t)
    {
        if (t < 0.0f)
            return;
        all.push_back(Timed{c, t});
        if (c.kind != SELL_KIND_LDGV3)
            return;
        const int p = c.cached ? 1 : 0;
        if (t < pol[p])
        {
            pol[p] = t;
            polc[p] = c;
        }
        int idx = 0;
        for (int v = c.parts; v > 1; v >>= 1)
            ++idx;
        if (idx >= 0 && idx < kParts && t < part[idx])
            part[idx] = t;
    }

    /*
      Always names the winner and quotes a positive margin. A signed delta would
      need the reader to remember which policy the sign is relative to, and the
      two lines sit next to a winner column that already uses %+.1f for
      something else.

      A second line gives LDG3 against every other family, signed as
      (LDG3 - family) / family, so positive means LDG3 is slower by that much
      and negative means it is faster. The winner line already carries each
      family's time, but reading a gap off it means dividing two four-decimal
      numbers by eye, per family, per slot, and the answer to "can LDG3 replace
      this one" is the gap rather than the times.

      Comparing only against the slot winner would print nothing useful wherever
      LDG3 already wins, which is half the slots. Under a search restricted to
      LDG3 nothing else is timed and the line is omitted.
    */
    void print(bool with_part, const float* best) const
    {
        if (pol[0] >= kInf && pol[1] >= kInf)
            return;
        char sbuf[48] = "n/a", cbuf[48] = "n/a";
        if (pol[0] < kInf)
        {
            char cfg[40];
            FormatCfg(cfg, sizeof cfg, polc[0], with_part);
            snprintf(sbuf, sizeof sbuf, "%s=%.4f", cfg, pol[0]);
        }
        if (pol[1] < kInf)
        {
            char cfg[40];
            FormatCfg(cfg, sizeof cfg, polc[1], with_part);
            snprintf(cbuf, sizeof cbuf, "%s=%.4f", cfg, pol[1]);
        }
        printf("       LDG3 policy: stream %-24s cached %-24s", sbuf, cbuf);
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
            const int v3i = FamIdx(SELL_KIND_LDGV3);
            bool any_rival = false;
            for (int k = 0; k < kNumFam; ++k)
                if (k != v3i && best[k] < kInf)
                    any_rival = true;
            if (any_rival)
            {
                printf("       LDG3 vs:    ");
                for (int k = 0; k < kNumFam; ++k)
                    if (k != v3i && best[k] < kInf)
                        printf(" %s %+.2f%%", kFamName[k], 100.0 * (v3 - best[k]) / best[k]);
                printf("   (+ is LDG3 slower)\n");
            }
        }
        if (!with_part)
            return;
        bool any = false;
        for (int i = 0; i < kParts; ++i)
            if (part[i] < kInf)
                any = true;
        if (!any)
            return;
        printf("       LDG3 part:  ");
        for (int i = 0; i < kParts; ++i)
            if (part[i] < kInf)
                printf(" p%-2d %.4f", 1 << i, part[i]);
        printf("\n");
    }

    // Best time for one family at one launch shape, or kInf when that family
    // has nothing feasible there. LDG carries no rows-per-thread, so its w stays
    // 0 and it never matches a shape that has one; that prints as n/a, which is
    // the honest answer rather than a silent mismatch.
    float at(int kind, const SellConfig& shape) const
    {
        float bt = kInf;
        for (const Timed& e : all)
            if (e.c.kind == kind && e.c.blk == shape.blk && e.c.unroll == shape.unroll && e.c.w == shape.w
                && e.t < bt)
                bt = e.t;
        return bt;
    }

    /*
      When another family takes a slot, two questions are tangled: is its launch
      shape better, or is its memory path better? LDG3, TMA and TMA2D read
      blk/unroll/rows-per-thread the same way, and all three advance 2*UNROLL
      k-steps per iteration, so a shape transfers between them unchanged. Timing
      each family at both shapes separates the two.

      What this cannot show: the sweep already minimises over LDG3, so LDG3 at
      the rival's shape is never faster than LDG3's own pick, and a near-tie
      there would be arithmetic, not evidence. The informative halves are how
      much LDG3 loses when forced into the rival's shape, which prices the
      shape, and how the rival fares at LDG3's shape, which prices the memory
      path with the geometry held fixed.
    */
    void print_geometry(const SellConfig& w, float wt, const SellConfig& v3c, float v3t) const
    {
        if (w.kind == SELL_KIND_LDGV3 || wt >= kInf || v3t >= kInf)
            return;
        const float v3_there = at(SELL_KIND_LDGV3, w);
        const float riv_here = at(w.kind, v3c);
        // LDG is templated on block and unroll only, so it shares no shape with
        // LDG3 and both halves come back empty. That line says nothing.
        if (v3_there >= kInf && riv_here >= kInf)
            return;
        const char* rival = kFamName[FamIdx(w.kind)];
        printf("       geometry:    at %d/%d/%d %s %.4f LDG3 ", w.blk, w.unroll, w.w, rival, wt);
        if (v3_there < kInf)
            printf("%.4f (%+.2f%%)", v3_there, 100.0 * (v3_there - wt) / wt);
        else
            printf("n/a");
        printf("  |  at %d/%d/%d LDG3 %.4f %s ", v3c.blk, v3c.unroll, v3c.w, v3t, rival);
        if (riv_here < kInf)
            printf("%.4f (%+.2f%%)", riv_here, 100.0 * (riv_here - v3t) / v3t);
        else
            printf("n/a");
        printf("\n");
    }
};

/*
  HPCG_FORCE_KIND, HPCG_FORCE_KIND_MV and HPCG_FORCE_KIND_SV restrict the search
  to the families named, as a comma-separated list: "2" searches LDG_V2 alone
  and "2,4" searches LDG_V2 and LDG3. A list rather than a single value because
  the question is usually "how do these two compare at each level", and one run
  answers it where two runs answer it against different GPU clocks.

  Naming a family this build does not contain is an error worth saying out loud
  rather than an empty search: the one number that does that, 5, is a real family
  in the tree this was ported from, so asking for it is a reasonable mistake with
  a misleading outcome.
*/
struct KindFilter
{
    bool active = false;
    bool allow[kNumFam] = {};
    // Which variable supplied it, so the log names the one that was actually
    // read rather than the one whose value it ended up carrying.
    const char* from = NULL;

    bool permits(int kind) const
    {
        if (!active)
            return true;
        const int i = FamIdx(kind);
        return i >= 0 && allow[i];
    }
};

KindFilter ParseKindFilter(const char* name, int rank)
{
    KindFilter f;
    const char* v = std::getenv(name);
    if (!v || !*v)
        return f;

    // Set before parsing, so a value that names only families this build does
    // not have restricts the search to nothing rather than leaving it
    // unrestricted. "Sweep every family" and "sweep the family I asked for" are
    // different experiments, and the second failing must not silently become
    // the first.
    f.active = true;
    f.from = name;
    const char* s = v;
    while (*s)
    {
        char* end = NULL;
        const long kind = std::strtol(s, &end, 10);
        if (end == s)
            break;
        const int i = FamIdx((int) kind);
        if (i < 0)
        {
            if (rank == 0)
                fprintf(stderr,
                    "[autotune] %s names family %ld, which this build does not contain; ignoring it. This build has "
                    "%d (LDG), %d (TMA), %d (LDG_V2), %d (TMA2D) and %d (LDG3).\n",
                    name, kind, SELL_KIND_LDG, SELL_KIND_TMA, SELL_KIND_LDGV2, SELL_KIND_TMA2D, SELL_KIND_LDGV3);
        }
        else
            f.allow[i] = true;
        s = end;
        while (*s == ',' || *s == ' ')
            ++s;
    }
    return f;
}

void PrintFilter(const char* op, const KindFilter& f)
{
    if (!f.active)
        return;
    printf("[autotune] %s: restricting the %s search to", f.from ? f.from : "?", op);
    bool any = false;
    for (int i = 0; i < kNumFam; ++i)
        if (f.allow[i])
        {
            printf(" %s", kFamName[i]);
            any = true;
        }
    if (!any)
        printf(" nothing this build contains");
    printf("\n");
}

// --- candidate generation -------------------------------------------------
//
// The lists are the ones the autotuner this was ported from swept, minus the two
// families that do not exist here, and they match what each launcher is actually
// instantiated for: asking for anything outside them would only produce a launch
// that returns false.

void AddLdg(std::vector<SellConfig>& v, bool mv)
{
    const int blk[] = {64, 128, 256};
    // Unroll 1 is an SpMV shape only; the solve is not instantiated for it.
    const int un_mv[] = {1, 4, 6, 7, 8, 10, 12, 14, 16};
    const int un_sv[] = {4, 6, 7, 8, 10, 12, 14, 16};
    const int* un = mv ? un_mv : un_sv;
    const int nun = mv ? 9 : 8;
    for (int b : blk)
        for (int i = 0; i < nun; ++i)
        {
            SellConfig c;
            c.kind = SELL_KIND_LDG;
            c.blk = b;
            c.unroll = un[i];
            v.push_back(c);
        }
}

void AddTma(std::vector<SellConfig>& v, bool mv)
{
    // The solve reaches one row block further than the SpMV does -- block 512
    // and 16 rows per thread -- because a colour is a fraction of the rows and
    // the block still has to divide it.
    const int blk_mv[] = {32, 64, 128, 256};
    const int blk_sv[] = {32, 64, 128, 256, 512};
    const int un[] = {1, 2, 3, 4, 6, 8};
    const int rpt_mv[] = {1, 2, 4, 8};
    const int rpt_sv[] = {1, 2, 4, 8, 16};
    const int nblk = mv ? 4 : 5;
    const int nrpt = mv ? 4 : 5;
    for (int i = 0; i < nblk; ++i)
        for (int u : un)
            for (int j = 0; j < nrpt; ++j)
            {
                SellConfig c;
                c.kind = SELL_KIND_TMA;
                c.blk = mv ? blk_mv[i] : blk_sv[i];
                c.unroll = u;
                c.w = mv ? rpt_mv[j] : rpt_sv[j];
                v.push_back(c);
            }
}

/*
  TMA2D's grid, the same for both operators.

  It is narrower than TMA's on two axes and for one reason each. Rows per thread
  stops at 4 because a thread's rows are consecutive here and are read and
  written as one vector access, and four doubles is the widest the hardware has.
  The block size stops at 256 where TMA's solve reaches 512, because the row
  block is what one tensor box covers rather than what a colour allows.
*/
void AddTma2d(std::vector<SellConfig>& v)
{
    const int blk[] = {32, 64, 128, 256};
    const int un[] = {1, 2, 3, 4, 6, 8};
    const int rpt[] = {1, 2, 4};
    for (int b : blk)
        for (int u : un)
            for (int r : rpt)
            {
                SellConfig c;
                c.kind = SELL_KIND_TMA2D;
                c.blk = b;
                c.unroll = u;
                c.w = r;
                v.push_back(c);
            }
}

// The SpMV row-partition counts. parts = 1 is the flat 1D walk, which is what
// the coarse levels appear to want; 8 is the value LDG hard-coded to spread the
// column stream over the memory system, which pays while the matrix is far
// larger than cache; 16 and 32 test the other direction at level 0, where
// nothing had checked whether 8 was the right choice. It is a launch shape
// rather than a code path, so every value costs one kernel argument and no
// extra instantiations.
const int kMvParts[] = {1, 2, 4, 8, 16, 32};

void AddLdgV2(std::vector<SellConfig>& v, bool mv)
{
    const int blk[] = {32, 64, 128, 256};
    const int un[] = {1, 2, 3, 4};
    const int w[] = {1, 2, 4, 8};
    for (int b : blk)
        for (int u : un)
            for (int ww : w)
            {
                SellConfig c;
                c.kind = SELL_KIND_LDGV2;
                c.blk = b;
                c.unroll = u;
                c.w = ww;
                if (!mv)
                {
                    v.push_back(c);
                    continue;
                }
                for (int p : kMvParts)
                {
                    c.parts = p;
                    v.push_back(c);
                }
            }
}

/*
  LDG3 starts from LDG_V2's blk/unroll/W grid and extends it on three axes:
  access width, cache policy, and unroll depth at small W. All three describe how
  W is fetched rather than naming a different kernel, so all three live here and a
  search restricted to LDG3 searches all of them.

  Access width: wide asks for the widest load available rather than 128-bit
  accesses. It only means something at 4 and 8 rows per thread, since 1 and 2 are
  already 8 and 16 bytes, so it is offered there and nowhere else.

  Cache policy: cached asks for ordinary cached loads instead of the streaming
  ones. This is the axis that decides LDG3 against LDG_V2. LDG3 took its policy
  from LDG, which streams, and LDG_V2 keeps the line; measurement put LDG_V2
  ahead at every level 3 slot and LDG3 ahead at levels 0 and 1, which is what a
  working set of 46 MiB against 23 GiB predicts. Making the policy a candidate
  rather than a property is what lets one family cover both.
*/
void AddLdgV3(std::vector<SellConfig>& v, bool mv)
{
    const int blk[] = {32, 64, 128, 256};
    const int un[] = {1, 2, 3, 4};

    // Depth at small W. LDG takes level 2 at 128/14 and 256/10 with one row per
    // thread, and LDG_V2's grid stops at unroll 4, so LDG3 inherited a cap that
    // made those configurations unreachable rather than merely unchosen. The unit
    // differs between the two: LDG advances Unroll k-steps per loop iteration
    // where LDG3's A/B pair advances 2*UNROLL, so LDG's 14 and 10 are unroll 7
    // and 5 here, and the lists bracket both. Depth costs 2*UNROLL*W*12 bytes of
    // register file, which is why it is offered at W of 1 and 2 and nowhere else
    // -- at W of 8 the existing cap of 4 already sits near 190 registers. No
    // wide, which needs W >= 4.
    const int deep_u1[] = {5, 6, 7, 8, 10, 14};
    const int deep_u2[] = {5, 6, 7, 8};

    // (w, wide) pairs, each swept over both cache policies.
    struct Shape
    {
        int w;
        bool wide;
    };
    std::vector<Shape> shapes;
    for (int ww : {1, 2, 4, 8})
        shapes.push_back(Shape{ww, false});
    for (int ww : {4, 8})
        shapes.push_back(Shape{ww, true});

    auto emit = [&](int b, int u, int ww, bool wide)
    {
        for (bool cached : {false, true})
        {
            SellConfig c;
            c.kind = SELL_KIND_LDGV3;
            c.blk = b;
            c.unroll = u;
            c.w = ww;
            c.wide = wide;
            c.cached = cached;
            if (!mv)
            {
                v.push_back(c);
                continue;
            }
            for (int p : kMvParts)
            {
                c.parts = p;
                v.push_back(c);
            }
        }
    };

    for (int b : blk)
    {
        for (int u : un)
            for (const Shape& s : shapes)
                emit(b, u, s.w, s.wide);
        for (int u : deep_u1)
            emit(b, u, 1, false);
        for (int u : deep_u2)
            emit(b, u, 2, false);
    }
}

std::vector<SellConfig> BuildCandidates(bool mv, const KindFilter& filter, int count[kNumFam])
{
    std::vector<SellConfig> all;
    if (filter.permits(SELL_KIND_LDG))
        AddLdg(all, mv);
    if (filter.permits(SELL_KIND_TMA))
        AddTma(all, mv);
    if (filter.permits(SELL_KIND_LDGV2))
        AddLdgV2(all, mv);
    if (filter.permits(SELL_KIND_TMA2D))
        AddTma2d(all);
    if (filter.permits(SELL_KIND_LDGV3))
        AddLdgV3(all, mv);
    for (int i = 0; i < kNumFam; ++i)
        count[i] = 0;
    for (const SellConfig& c : all)
    {
        const int i = FamIdx(c.kind);
        if (i >= 0)
            ++count[i];
    }
    return all;
}

// --- one slot -------------------------------------------------------------

struct SlotResult
{
    bool feasible = false;
    SellConfig win;
    float wint = kInf;
};

/*
  Time every candidate at one level for one operator, print the row, and return
  the winner. `tag` is empty for the two selecting sweeps and "MV-L " / "MV-U "
  for the diagnostic ones; the leading-whitespace-then-tag-then-level shape of
  the printed line is what slotlib.py matches a slot on.
*/
template <class TimeFn>
SlotResult SweepSlot(const char* tag, const SparseMatrix& m, const std::vector<SellConfig>& cands, bool with_parts,
    double bytes, int rank, int iters, TimeFn&& timer)
{
    float best[kNumFam];
    SellConfig bestc[kNumFam];
    for (int k = 0; k < kNumFam; ++k)
    {
        best[k] = kInf;
        bestc[k] = SellConfig{};
        bestc[k].kind = kFamKind[k];
    }
    Ldg3Split split;
    split.reset();

    for (const SellConfig& c : cands)
    {
        const float t = timer(c);
        if (t < 0.0f)
            continue;
        split.add(c, t);
        const int k = FamIdx(c.kind);
        if (k >= 0 && t < best[k])
        {
            best[k] = t;
            bestc[k] = c;
        }
    }

    int win = 0;
    for (int k = 1; k < kNumFam; ++k)
        if (best[k] < best[win])
            win = k;

    SlotResult out;
    if (best[win] >= kInf)
    {
        if (rank == 0)
            printf("  %sL%d rows=%-9d | no feasible config (skipped)\n", tag, m.level, (int) m.localNumberOfRows);
        return out;
    }
    out.feasible = true;
    out.win = bestc[win];
    out.wint = best[win];

    if (rank != 0)
        return out;

    char buf[kNumFam][40], winbuf[48];
    for (int k = 0; k < kNumFam; ++k)
    {
        if (best[k] >= kInf)
            snprintf(buf[k], sizeof buf[k], "n/a");
        else
        {
            char cfg[32];
            FormatCfg(cfg, sizeof cfg, bestc[k], with_parts);
            snprintf(buf[k], sizeof buf[k], "%s=%.4f", cfg, best[k]);
        }
    }
    {
        char cfg[32];
        FormatCfg(cfg, sizeof cfg, bestc[win], with_parts);
        snprintf(winbuf, sizeof winbuf, "%s %s", kFamName[win], cfg);
    }
    float second = kInf;
    for (int k = 0; k < kNumFam; ++k)
        if (k != win && best[k] < second)
            second = best[k];
    const double margin = (second < kInf) ? 100.0 * (second - best[win]) / second : 0.0;

    char line[kNumFam * 48];
    int off = 0;
    for (int k = 0; k < kNumFam; ++k)
        off += snprintf(line + off, sizeof line - off, "%s %-18s ", kFamName[k], buf[k]);
    printf("  %sL%d rows=%-9d | %s| WIN %-16s (%+.1f%%)", tag, m.level, (int) m.localNumberOfRows, line, winbuf,
        margin);
    if (bytes > 0.0)
        printf(" BW=%.0f", bytes / best[win] / 1.0e6);
    // Per level now, so a row can be read for how well its margin is resolved:
    // a +0.6% win at 3 iterations and the same margin at 32 are not the same
    // claim. Appended rather than inserted, because slotlib.py matches a slot
    // on the front of the line.
    printf(" it=%d", iters);
    printf("\n");
    split.print(with_parts, best);
    split.print_geometry(bestc[win], best[win], bestc[FamIdx(SELL_KIND_LDGV3)], best[FamIdx(SELL_KIND_LDGV3)]);
    return out;
}

/*
  Iterations for one level, chosen so that every level gets a comparable
  measurement budget rather than a comparable iteration count.

  A fixed count spends the time where it is least needed. At level 0 a SymGS
  config takes about 4 ms and three of them settle the answer well clear of the
  next family. At level 2 a config takes 0.11 ms, every family lands within a
  couple of percent of every other, and three iterations put the whole decision
  inside the noise: across two runs of the same tree the level-2 winner changed
  family, and an unchanged tree's level-2 time moved 8%. The sweep was then
  choosing between configurations it could not tell apart, and reporting the
  choice as though it could.

  The budget is whatever the finest level already costs, so that level is timed
  exactly as before and the coarse ones are brought up to it. Cost per level
  stays roughly flat instead of falling away with the row count -- which is
  affordable precisely because the coarse levels are cheap: at level 2 the whole
  sweep spent under a second where level 0 spent seven.

  Capped, because level 3 is mostly launch latency rather than work, so matching
  the budget there would ask for hundreds of iterations of a kernel whose cost
  does not average down with more of them.

  The probe costs one config per level at the base count, and its own time only
  has to be right to an order of magnitude, so an unrepresentative first
  candidate cannot do worse than leave a level at the count it used before.
*/
template <class TimeFn>
int SlotIters(const std::vector<SellConfig>& cands, int base, int max_iters, double& budget_ms, TimeFn&& timer)
{
    for (const SellConfig& c : cands)
    {
        const float t = timer(c, base);
        if (t <= 0.0f)
            continue;
        if (budget_ms <= 0.0)
        {
            budget_ms = (double) base * t;
            return base;
        }
        int it = (int) (budget_ms / t + 0.5);
        if (it < base)
            it = base;
        if (it > max_iters)
            it = max_iters;
        return it;
    }
    return base;
}

// Bytes of matrix stream one apply of the operator moves, for the BW column: one
// column index and one value per stored entry. SymGS reads the two triangles,
// which between them hold every entry but the diagonal.
double MvBytes(const SparseMatrix& m)
{
    const double entry = (double) (sizeof(local_int_t) + sizeof(double));
    return (double) m.localNumberOfNonzeros * entry;
}

double SvBytes(const SparseMatrix& m)
{
    const double entry = (double) (sizeof(local_int_t) + sizeof(double));
    return (double) (m.localNumberOfNonzeros - m.localNumberOfRows) * entry;
}

void ReportChoice(const char* what, int level, const SellConfig& c, bool with_parts)
{
    char cfg[32];
    FormatCfg(cfg, sizeof cfg, c, with_parts);
    // No leading whitespace, so slotlib.py cannot mistake a summary line for a
    // slot row.
    printf("[autotune] chose %s level %d: %s %s\n", what, level, kFamName[FamIdx(c.kind)], cfg);
}
} // namespace

void AutotuneSymGS(const SparseMatrix& A_top)
{
    const char* en = std::getenv("HPCG_AUTOTUNE");
    if (!en || std::atoi(en) == 0)
        return;

    const int rank = A_top.geom ? A_top.geom->rank : 0;

    // 10 iterations rather than the 20 this was ported with. The candidate space
    // is the whole of it -- five families over block size, unroll and W, plus
    // LDG3's width, policy and partition axes, which is where the family's
    // advantage lives -- so the space is not the thing to trim; the sample count
    // is. HPCG_AUTOTUNE_ITERS overrides it.
    int iters = 10;
    if (const char* it = std::getenv("HPCG_AUTOTUNE_ITERS"))
        if (*it)
            iters = std::atoi(it);

    // Ceiling on what SlotIters may raise a coarse level to. 32 rather than the
    // budget's own answer, which at level 3 runs to the hundreds for a kernel
    // whose cost is mostly launch latency and does not average down. At 32 a
    // coarse sweep costs a second or two against level 0's seven, and the
    // sampling error falls by about three -- enough to resolve the couple of
    // percent that separates the families there, which 3 could not.
    int max_iters = 32;
    if (const char* it = std::getenv("HPCG_AUTOTUNE_MAX_ITERS"))
        if (*it)
            max_iters = std::atoi(it);
    if (max_iters < 1)
        max_iters = 1;
    if (iters < 1)
        iters = 1;

    /*
      Per-operator overrides. A family need not serve both operators equally
      well, and restricting one of them globally would leave the other with an
      empty candidate list, whereupon every level reports "no feasible config"
      and quietly keeps its heuristic pick. That reads as a single-family run but
      is a blend.
    */
    const KindFilter force = ParseKindFilter("HPCG_FORCE_KIND", rank);
    KindFilter force_mv = ParseKindFilter("HPCG_FORCE_KIND_MV", rank);
    KindFilter force_sv = ParseKindFilter("HPCG_FORCE_KIND_SV", rank);
    if (!force_mv.active)
        force_mv = force;
    if (!force_sv.active)
        force_sv = force;

    int n_sv[kNumFam], n_mv[kNumFam];
    const std::vector<SellConfig> svcands = BuildCandidates(false, force_sv, n_sv);
    const std::vector<SellConfig> mvcands = BuildCandidates(true, force_mv, n_mv);

    // A pin applies one configuration to every level and outranks anything
    // chosen here, so sweeping under one would spend the time and then discard
    // the answer.
    const bool mv_pinned = std::getenv("HPCG_PIN_MV") != NULL && *std::getenv("HPCG_PIN_MV") != '\0';
    const bool sv_pinned = std::getenv("HPCG_PIN_SV") != NULL && *std::getenv("HPCG_PIN_SV") != '\0';

    const bool sweep_dirs = [] {
        const char* e = std::getenv("HPCG_TUNE_DIRS");
        return e && *e && std::atoi(e) != 0;
    }();

    if (rank == 0)
    {
        PrintFilter("SymGS", force_sv);
        PrintFilter("SpMV", force_mv);
        // An empty list is not the same as a family that loses; say so here
        // rather than leaving four "no feasible config" lines to be interpreted.
        if (svcands.empty())
            printf("[autotune] WARNING: no SymGS candidates after filtering; SymGS keeps its g_config pick\n");
        if (mvcands.empty())
            printf("[autotune] WARNING: no SpMV candidates after filtering; SpMV keeps its g_config pick\n");
        if (sv_pinned)
            printf("[autotune] HPCG_PIN_SV pins every level; skipping the SymGS search\n");
        if (mv_pinned)
            printf("[autotune] HPCG_PIN_MV pins every level; skipping the SpMV search\n");
    }

    const size_t n = (size_t) A_top.localNumberOfColumns;
    double* rv = NULL;
    double* xv = NULL;
    if (cudaMalloc((void**) &rv, n * sizeof(double)) != cudaSuccess
        || cudaMalloc((void**) &xv, n * sizeof(double)) != cudaSuccess)
    {
        if (rank == 0)
            fprintf(stderr, "[autotune] scratch alloc failed; skipping autotune\n");
        cudaFree(rv);
        cudaFree(xv);
        cudaGetLastError();
        return;
    }
    cudaMemset(rv, 0, n * sizeof(double));
    cudaMemset(xv, 0, n * sizeof(double));

    if (!sv_pinned && !svcands.empty())
    {
        if (rank == 0)
            printf("\n===== HPCG SymGS (SV) autotune: %d LDG + %d TMA + %d LDG_V2 + %d TMA2D + %d LDG3 "
                   "configs/level, iters=%d..%d per level =====\n",
                n_sv[0], n_sv[1], n_sv[2], n_sv[3], n_sv[4], iters, max_iters);

        double budget = 0.0;
        for (const SparseMatrix* m = &A_top; m != NULL; m = m->Ac)
        {
            const int it = SlotIters(svcands, iters, max_iters, budget,
                [&](const SellConfig& c, int n) { return TimeSvConfig(*m, rv, xv, c, n); });
            const SlotResult r = SweepSlot("", *m, svcands, false, SvBytes(*m), rank, it,
                [&](const SellConfig& c) { return TimeSvConfig(*m, rv, xv, c, it); });
            if (r.feasible)
                SetSvChoice(m->level, r.win);
        }
    }

    if (!mv_pinned && !mvcands.empty())
    {
        if (rank == 0)
            printf("\n===== HPCG SpMV (MV) autotune: %d LDG + %d TMA + %d LDG_V2 + %d TMA2D + %d LDG3 "
                   "configs/level, iters=%d..%d per level =====\n",
                n_mv[0], n_mv[1], n_mv[2], n_mv[3], n_mv[4], iters, max_iters);

        double budget = 0.0;
        for (const SparseMatrix* m = &A_top; m != NULL; m = m->Ac)
        {
            const int it = SlotIters(mvcands, iters, max_iters, budget,
                [&](const SellConfig& c, int n) { return TimeMvConfig(*m, xv, rv, c, n); });
            const SlotResult r = SweepSlot("", *m, mvcands, true, MvBytes(*m), rank, it,
                [&](const SellConfig& c) { return TimeMvConfig(*m, xv, rv, c, it); });
            if (r.feasible)
                SetMvChoice(m->level, r.win);
        }
    }

    /*
      Diagnostic only, off by default. The MV winner above was chosen by timing
      the full matrix A, but that one choice is then used for the L and U
      multiplies inside SymGS too -- half the nonzeros each, and L accumulates
      where A and U overwrite. Those two run in the MG hot path and are never
      measured. This sweeps each family against them and prints what would have
      been picked. Nothing here changes the selection, so every arm keeps its
      meaning and the extra sweep time is only spent when asked for.
    */
    if (sweep_dirs && !mvcands.empty())
    {
        for (int dir = 1; dir <= 2; ++dir)
        {
            const DIR d = (dir == 1) ? Forward : Backward;
            if (rank == 0)
                printf("\n===== MV on the %s submatrix, diagnostic only, selection unchanged =====\n",
                    (dir == 1) ? "L (SymGS forward, beta=1, accumulates)" : "U (SymGS backward, beta=0, overwrites)");
            double budget = 0.0;
            for (const SparseMatrix* m = &A_top; m != NULL; m = m->Ac)
            {
                const int it = SlotIters(mvcands, iters, max_iters, budget,
                    [&](const SellConfig& c, int n) { return TimeMvConfigDir(*m, xv, rv, c, n, d); });
                SweepSlot(dir == 1 ? "MV-L " : "MV-U ", *m, mvcands, true, 0.0, rank, it,
                    [&](const SellConfig& c) { return TimeMvConfigDir(*m, xv, rv, c, it, d); });
            }
        }
    }

    cudaFree(rv);
    cudaFree(xv);
    cudaGetLastError();

    /*
      The point of the whole file: what each level ended up with. Printed
      together rather than left to be read off four slot rows, because "did the
      selection actually vary by level" is the question this answers, and one
      configuration repeated four times means either the plumbing is not being
      consulted per level or the search is degenerate.
    */
    if (rank == 0)
    {
        printf("\n");
        // Read back out of the arrays mv_sell / sv_sell themselves consult,
        // rather than out of a copy kept here, so this reports what will
        // actually run and not what the search believed it installed.
        for (const SparseMatrix* m = &A_top; m != NULL; m = m->Ac)
        {
            SellConfig c;
            if (GetSvChoiceForLevel(m->level, c))
                ReportChoice("SymGS", m->level, c, false);
        }
        for (const SparseMatrix* m = &A_top; m != NULL; m = m->Ac)
        {
            SellConfig c;
            if (GetMvChoiceForLevel(m->level, c))
                ReportChoice("SpMV", m->level, c, true);
        }
        printf("=========================================================================\n\n");
    }
}

#endif
