# Artifact QA notes

## Word report

- The authoritative course DOCX template is ISO/IEC Strict OOXML and remains byte-for-byte unchanged outside this delivery folder (SHA-256: `B91FFEF155007F703314EAC8811AD363C69A1425C407B1B66C7C72E1AAF4FB67`).
- A Transitional working copy was used only because `python-docx` cannot open the Strict namespace directly; no template parts were removed, and the final file adds only its comparison image media part.
- Final structural checks cover five inherited sections, A4 geometry, heading hierarchy, exact table widths, table header semantics, image inventory/alt text, source placeholders, and general accessibility.
- LibreOffice is unavailable on this machine and Microsoft Word COM returns `CO_E_SERVER_EXEC_FAILURE`, so the DOCX cannot be raster-rendered here. Structural audits are recorded under `reproducibility/qa/`; the whole-group checklist therefore retains one final open-and-view check on the submission machine.

## PowerPoint slide

- The slide duplicates the supplied course `Results` slide and edits only the planned title, body, page number, presenter footer and speaker notes.
- The final slide is directly rendered to PNG for human inspection, and the template-fidelity checker must report zero issues.
- The generic `slides_test.py` padding check cannot complete because the supplied template package contains an unsupported EMF resource; its artifact-tool subprocess still produces the PNG. Direct rendering, layout bounds and the dedicated template-fidelity audit are used as the authoritative checks.

## Identity and group integration

The source materials do not provide the actual Member 3 name or zID. The contribution therefore uses the neutral presenter label `Member 3 baseline lead` rather than inventing an identity. Replace it only when merging into the whole-group presentation.
