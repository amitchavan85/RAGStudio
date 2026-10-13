import { describe, expect, it } from "vitest";
import { ApiError, type ModelOut } from "../api/client";
import { ACCENTS, BASE, contrast, type Accent, type Mode } from "../theme";
import { answerToMarkdown, costPerAnswer, friendlyError, matchItems, modelForTier, splitCitations } from "./logic";

const m = (name: string, tier: string, inp: number, out: number, available = true): ModelOut => ({
  name, provider: "openai", tier, input_per_m: inp, output_per_m: out, available, reason: available ? null : "no key",
});

describe("splitCitations", () => {
  it("turns known [n] markers into citations and keeps the rest as text", () => {
    expect(splitCitations("India grew 8.2% [1] and the US 2.5% [2].", new Set([1, 2]))).toEqual([
      { kind: "text", text: "India grew 8.2% " },
      { kind: "cite", n: 1 },
      { kind: "text", text: " and the US 2.5% " },
      { kind: "cite", n: 2 },
      { kind: "text", text: "." },
    ]);
  });
  it("leaves unknown numbers like [7] or [2023] as plain text", () => {
    expect(splitCitations("See [7] in [2023].", new Set([1]))).toEqual([{ kind: "text", text: "See [7] in [2023]." }]);
  });
});

describe("modelForTier", () => {
  const models = [m("gpt-5-nano", "economy", 0.05, 0.4), m("gpt-4o-mini", "economy", 0.15, 0.6),
    m("claude-haiku-5-5", "economy", 0.1, 0.5, false), m("gpt-5.4", "premium", 2.5, 15)];
  it("prefers the server default when it is in the tier", () => {
    expect(modelForTier(models, "economy", "gpt-4o-mini")?.name).toBe("gpt-4o-mini");
  });
  it("otherwise picks the cheapest available model", () => {
    expect(modelForTier(models, "economy", "gpt-5.4")?.name).toBe("gpt-5-nano");
  });
  it("skips unavailable models and empty tiers", () => {
    expect(modelForTier([m("claude-haiku-5-5", "economy", 0.1, 0.5, false)], "economy")).toBeUndefined();
    expect(modelForTier(models, "local")).toBeUndefined();
  });
});

describe("costPerAnswer", () => {
  it("estimates per-answer cost and says free for local models", () => {
    expect(costPerAnswer(m("gpt-4o-mini", "economy", 0.15, 0.6))).toBe("~$0.00028");
    expect(costPerAnswer(m("gpt-5.5", "premium", 5, 30))).toBe("~$0.011");
    expect(costPerAnswer(m("llama3.2:3b", "local", 0, 0))).toBe("free");
  });
});

describe("friendlyError", () => {
  it("maps API codes to human messages", () => {
    expect(friendlyError(new ApiError(429, "rate_limited", "x", { retry_after_seconds: 12 }))).toContain("12 seconds");
    expect(friendlyError(new ApiError(502, "provider_error", "x"))).toContain("Try another model");
    expect(friendlyError(new ApiError(0, "network_error", "x"))).toContain("Can't reach the server");
    expect(friendlyError(new Error("boom"))).toBe("Something went wrong. Please try again.");
  });
});

describe("answerToMarkdown", () => {
  it("adds a reference list", () => {
    expect(answerToMarkdown("Q?", "A [1].", [{ n: 1, source: "a.pdf", page: 3 }])).toBe("**Q:** Q?\n\nA [1].\n\n[1] a.pdf, page 3\n");
  });
});

describe("theme contrast (WCAG AA, 4.5:1 for normal text)", () => {
  const modes: Mode[] = ["light", "dark"];
  for (const accent of Object.keys(ACCENTS) as Accent[]) {
    for (const mode of modes) {
      it(`${accent} / ${mode}`, () => {
        const t = { ...BASE[mode], ...ACCENTS[accent][mode] };
        const pairs: [string, string, string][] = [
          ["text on bg", t.text, t.bg], ["text on surface", t.text, t.surface],
          ["muted on bg", t["text-muted"], t.bg], ["muted on surface", t["text-muted"], t.surface],
          ["accent on surface", t.accent, t.surface], ["button text on accent", t["on-accent"], t.accent],
          ["citation text on tint", t["tint-text"], t.tint],
          ["success", t.success, t.surface], ["warning", t.warning, t.surface], ["danger", t.danger, t.surface],
        ];
        for (const [name, fg, bg] of pairs) expect(contrast(fg, bg), name).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
});

describe("matchItems (highlight the cited passage on the PDF page)", () => {
  const page = ["United States — Macroeconomic & GDP Report", "Year", "Nominal GDP (USD bn)", "2022", "25,460", "+2.1%",
    "2023", "27,360", "+2.5%", "Outlook", "Growth is expected to remain moderate, supported by resilient", "consumer spending."];
  it("marks the items of a table row the chunk quotes (separators differ)", () => {
    const hit = matchItems(page, "... 2022 | 25,460 | +2.1% 2023 | 27,360 | +2.5% ...");
    expect([...hit].sort((a, b) => a - b)).toEqual([3, 4, 5, 6, 7, 8]);
  });
  it("matches prose across line breaks and ignores lone common words", () => {
    const hit = matchItems(page, "Growth is expected to remain moderate, supported by resilient\nconsumer spending.");
    expect([...hit].sort((a, b) => a - b)).toEqual([10, 11]);
    expect(matchItems(page, "the GDP of the").size).toBe(0);
  });
  it("ignores a common phrase that recurs elsewhere on the page (precision)", () => {
    const twoColumn = ["Growth in the euro area is expected to recover to", "1.3 percent over 2027-28, slightly outpacing",
      "announced measures is expected to be positive,", "and exports. Over 2027-28, growth is anticipated"];
    const hit = matchItems(twoColumn, "Growth in the euro area is expected to recover to 1.3 percent over 2027-28, slightly outpacing");
    expect([...hit].sort((a, b) => a - b)).toEqual([0, 1]);
  });
  it("still matches a passage shorter than the minimum run", () => {
    expect([...matchItems(["Capital", "Washington, D.C.", "Currency"], "Washington, D.C.")]).toEqual([1]);
  });
  it("handles empty input", () => {
    expect(matchItems(page, "").size).toBe(0);
    expect(matchItems([], "Growth is expected").size).toBe(0);
  });
});
