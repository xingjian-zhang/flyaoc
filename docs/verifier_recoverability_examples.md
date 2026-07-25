# Corpus-grounding verifier — worked recoverability examples (spot-check)

FLYAOC restricts its **primary** evaluation to *corpus-grounded* annotations: labels whose evidence
is a specific sentence in a specific open-access paper in the released corpus. Recoverability is a
**bounded** check — given a *known* target annotation and the cited paper, decide only whether the
text supports that annotation. The verifier never searches for or invents labels.

This file lets you verify that in seconds. For each example: **open the paper link, read the one
quoted sentence, and judge for yourself.** Each decision was produced by an automated verifier
(GPT-5) and then confirmed by a domain expert (a co-author).

## Summary — open a paper and check

| Gene | Annotation to verify | Cited paper | Verifier | Expert |
|---|---|---|---|---|
| *Ten-a* | expressed in the **antennal lobe** | [PMC3345284](https://pmc.ncbi.nlm.nih.gov/articles/PMC3345284/) | Supported | ✓ Confirmed |
| *lola* | expressed in **primary spermatocytes** | [PMC4006830](https://pmc.ncbi.nlm.nih.gov/articles/PMC4006830/) | Supported | ✓ Confirmed |
| *lola* | expressed in the **germinal proliferation center** | [PMC4006830](https://pmc.ncbi.nlm.nih.gov/articles/PMC4006830/) | Supported | ✓ Confirmed |

---

## 1. *Ten-a* — expressed in the antennal lobe  (FBbt:00003924)

**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3345284/

**Supporting sentence (verbatim):**
> "Both Drosophila Teneurins were endogenously expressed in the developing antennal lobe."

Context in the same paper: "…~50 discrete glomeruli in the antennal lobe"; a figure shows "antennal
lobes … stained by antibodies against Ten-m, Ten-a, and a neuropil marker, N-cadherin."

**Automated verifier (GPT-5):** *Supported.* The antennal lobe is named directly as the expression
site — both Drosophila Teneurins (Ten-a is one) are reported expressed in the developing antennal
lobe, with glomerular antibody staining. The label is directly recoverable from the text.

**Domain-expert verdict (co-author):** **Confirmed.** Ten-a is a Teneurin, and the paper explicitly
reports Ten-a expression in the antennal lobe. A non-specialist can confirm this from the quoted
sentence alone.

## 2. *lola* — expressed in primary spermatocytes  (FBbt:00005286)

**Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC4006830/

**Supporting sentence (verbatim):**
> "…both isoforms were expressed throughout the germinal proliferation center and primary spermatocytes."

Context: "lola is … required for … the switch … to spermatocyte growth and differentiation," shown by
RNA in situ hybridization.

**Automated verifier (GPT-5):** *Supported.* In situ hybridization localizes lola transcripts to
primary spermatocytes, and lola is required for the spermatogonia-to-spermatocyte transition. The
label is recoverable.

**Domain-expert verdict (co-author):** **Confirmed.** RNA in situ directly places lola transcripts in
primary spermatocytes; the expression annotation is unambiguous.

## 3. *lola* — expressed in the germinal proliferation center  (FBbt:00005259)

Same paper and same sentence as #2 ("…expressed throughout the germinal proliferation center and
primary spermatocytes"), illustrating that one cited paper can support more than one recoverable label.

**Verifier:** *Supported.* **Domain-expert verdict (co-author):** **Confirmed.**

---

**The check also rejects.** When the cited text does not support a candidate label, the verifier
returns *not recoverable* and the label is excluded from the primary evaluation set — the check
discriminates rather than rubber-stamps. A larger expert-audited sample, with inter-annotator
agreement and false-positive / false-negative rates, will be reported in a camera-ready appendix.
