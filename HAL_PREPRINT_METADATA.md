# HAL Preprint Metadata

**Title:**  
A Bilingual Latent Model of Proposition Hierarchy in Wittgenstein's Tractatus

**Author:**  
Mark A. Griffiths

**Affiliation:**  
Department of Academic Neuroscience, King's College London

**Document type:**  
Preprint / working paper

**Full-text language:**  
English

**Target journal:**  
Journal of Data Mining & Digital Humanities

**Preprint note:**  
This manuscript is a preprint prepared for deposit in HAL and intended for submission to the Journal of Data Mining & Digital Humanities.

**Abstract:**  
Wittgenstein's Tractatus Logico-Philosophicus (1922) is suited to structural modelling because its 526 propositions encode a tree-like scaffold of hierarchy depth, local families, sequential order and child structure. This paper uses that scaffold as weak supervision for learning compact latent representations of proposition position, and asks whether such representations align German and English realisations of the same proposition. The monolingual study uses English texts; the bilingual study uses German and English versions, yielding 1052 language-specific samples aligned by proposition ID.

The retained model is a split-latent variational autoencoder (VAE) with gated recurrent units (GRU) encoder and decoder. Its latent space separates a 24-dimensional text latent, used for reconstruction, from an 8-dimensional structure latent, used by structural heads predicting parent proposition, depth, next proposition and child count. In the bilingual model, the decoder is language-conditioned and an optional alignment loss pulls same-ID German and English structure latents together.

The monolingual seed sweep (10 seeds, regularisation at 0.05) shows that the split structure latent recovers hierarchy-relevant information. Parent accuracy is 0.6983 ± 0.0152, depth accuracy 0.9274 ± 0.0275 and next-proposition accuracy 0.3724 ± 0.0198, with reconstruction loss 1.2923 ± 0.0194 and perplexity 3.6416 ± 0.0706. Sibling propositions are closer in structure space (2.0656 ± 0.0342) than unrelated propositions (5.5574 ± 0.0532).

The bilingual sweep across ten seeds and five alignment strengths shows strong cross-language proposition tracking. With no alignment loss, Top-1 retrieval is 0.9348 ± 0.0115 and mean reciprocal rank (MRR) is 0.9650 ± 0.0066. Light alignment (lambda = 0.03) gives the strongest mean scores: Top-1 retrieval 0.9365 ± 0.0103, MRR 0.9662 ± 0.0058, parent accuracy 0.8608, depth accuracy 0.9643 and next accuracy 0.6189. High alignment (lambda = 1.00) degrades retrieval (0.7943 ± 0.0188) and MRR (0.8577 ± 0.0130). Shared hierarchy supervision therefore induces a compact proposition-level structure latent that aligns bilingual realisations without demonstrating general language-invariant logic or philosophical understanding of Wittgenstein's text.

**Keywords:**  
digital humanities; Tractatus Logico-Philosophicus; variational autoencoder; representation learning; textual hierarchy; bilingual alignment; computational philosophy; latent structure

**Licence recommendation:**  
CC BY 4.0, unless a later publisher agreement requires otherwise.

**Repository/code:**  
https://github.com/mark-alfred-griffiths/Tractatus_Logico_Philosophicus_Bilingual_Study
