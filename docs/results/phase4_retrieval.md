# Phase 4 — current information as cited evidence

Built 2026-10-11. Provider: Tavily Search API (`basic` depth, 1 credit per search, results cached 24 h).
The provider is swappable (`SearchProvider` protocol in `src/adapt/retrieval/providers.py`); tests use a
static provider with no network.

## Pipeline

For each value a criterion or hard limit needs but an option is missing:

1. **Search** `"<option> <attribute>"` (capped per run; cached).
2. **Candidates:** a number is taken only from a sentence on the page that names the attribute and the option;
   that sentence is kept as the quote. Numbers from the product name ("XPS 13") and bare years are skipped. If the
   attribute's name implies a unit (`battery_hours`, `weight_kg`), the number must be written with that unit, and
   "8 hours and 40 minutes" reads as 8.67.
3. **Source quality:** official (.gov, .edu, …), manufacturer, reference, general, user-generated (Reddit, Quora,
   forums: shown, never applied), rejected (not HTTPS).
4. **Freshness:** fast-changing attributes need a dated source within a limit (price 30 days, rent 90, salary 180,
   deadline 7, …); others need ≤ 3 years and may be undated (flagged).
5. **Decision per value:**
   - **apply:** fresh, usable, either two independent sites agree or an official/manufacturer site states it, and
     **no usable source states a different value**;
   - **confirm:** sources disagree, or only one ordinary site states it;
   - **none:** no usable source.
6. **Applied values** are written as `RETRIEVED` with URL, title, publication and retrieval dates. User values are
   never overwritten. Every decision lists the retrieved facts it used in `citations`.

## Live checks (2026-10-11, 5 credits)

| Option / attribute | Result | Correct? |
|---|---|---|
| MacBook Air M3 13" / weight_kg | **1.24 kg, applied** (everymac.com + support.apple.com agree) | ✅ matches Apple's spec |
| MacBook Air M3 13" / battery_hours | **confirm:** 15.22 h measured (Laptop Mag) vs. 18 h claimed (Amazon, Macworld) | ✅ a real disagreement (test vs. claim) |
| Dell XPS 13 9350 / battery_hours | **confirm:** 18, 18.57, 26 h from different reviews | ✅ different configurations; asking is right |
| Dell XPS 13 9350 / weight_kg | **none** | ✅ safe; the pages state pounds, not kg |

## Bugs found by the live run (fixed, with regression tests)

1. "8 hours and 40 minutes" was read as 8 → minutes now added.
2. Battery capacity ("52.6 Wh") was taken as battery life → unit must match the attribute.
3. A value was applied while the manufacturer's site said something else (only exact ties triggered "confirm") →
   **any** disagreeing usable source now means confirm.
4. "up to 13 hours" for the XPS **13** was skipped as part of the name → name numbers count when written with the unit.

## Known limitations

- No unit conversion (pounds → kg, inches → cm), so such values are missed rather than converted.
- Manufacturer detection uses the first word of the option name, so "MacBook" doesn't match apple.com and Apple's
  site is rated "general".
- Product variants (OLED vs. FHD screen) aren't distinguished; disagreement shows up as "confirm", which is safe but
  needs the user.
- Only 4 live lookups so far: these show the mechanics work, not a measured accuracy. A labelled set of real
  lookups is needed for that.
