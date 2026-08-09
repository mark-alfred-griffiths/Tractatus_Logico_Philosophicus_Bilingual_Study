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
Purpose: This article asks how far the proposition numbering of Wittgenstein’s Tractatus can serve as weak structural supervision for a bilingual, text-conditioned representation. The aim is validation-led Digital Humanities: to make supplied formal organization empirically inspectable, not to automate philosophical interpretation. 
Design/methodology/approach: Parent, depth, successor, and child-count targets are derived from proposition identifiers, while the encoder receives tokenized German and English text. A split-latent variational autoencoder is evaluated through retained-corpus diagnostics, seven ablation conditions, deterministic five-fold immediate-parent-family holdout, controlled paired-batch alignment, lexical references, and a frozen text-blind case-selection protocol. 
Findings: Retained-corpus structure-latent retrieval is high (Top-1 0.9360), but successor-only and shuffled-target ablations show that retrieval alone is weak evidence for hierarchy-derived organization. In held-out family evaluation, a small structure-latent retrieval signal remains (Top-1 0.0739), above reconstruction-only (0.0158) but below character term frequency–inverse document frequency (TF–IDF; 0.5619); exact parent and successor prediction are invalid for mostly unseen classes. Paired alignment tightens retained same-ID distances without establishing semantic equivalence. Frozen case studies identify family, bilingual-neighbourhood, and hierarchy–sequence prompts for close reading. 
Originality: The study combines weak supervision with ablation, held-out, alignment, and text-blind case-selection checks, showing what the representation supports and where retained retrieval can mislead. 
Contribution to the field of Digital Humanities: It provides a reproducible workflow for validating structurally supervised representations and using them to organize, rather than replace, scholarly interpretation.position tracking. With no alignment loss, Top-1 retrieval is 0.9348 ± 0.0115 and mean reciprocal rank (MRR) is 0.9650 ± 0.0066. Light alignment (lambda = 0.03) gives the strongest mean scores: Top-1 retrieval 0.9365 ± 0.0103, MRR 0.9662 ± 0.0058, parent accuracy 0.8608 ± 0.0094, depth accuracy 0.9643 ± 0.0136 and next accuracy 0.6189 ± 0.0219. High alignment (lambda = 1.00) degrades retrieval (0.7943 ± 0.0188) and MRR (0.8577 ± 0.0130). Shared hierarchy supervision therefore induces a compact proposition-level structure latent that aligns bilingual realisations without demonstrating general language-invariant logic or philosophical understanding of Wittgenstein's text.

**Keywords:**  
digital humanities; Tractatus Logico-Philosophicus; variational autoencoder; representation learning; textual hierarchy; bilingual alignment; computational philosophy; latent structure

**Licence recommendation:**  
CC BY 4.0, unless a later publisher agreement requires otherwise.

**Repository/code:**  
https://github.com/mark-alfred-griffiths/Tractatus_Logico_Philosophicus_Bilingual_Study
