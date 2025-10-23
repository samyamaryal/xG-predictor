# ETHICS.md

## Scope & Intent
- Academic research and teaching only; no commercial use.
- Aligns with **academic integrity and reproducibility standards**.

## Data Sources & Compliance
- Data sources (SoccerNet) are **open and public**.
- No **personally identifiable information (PII)** is collected or generated.
- All **data usage** complies with provider **Terms of Service**.
- All credentials are stored securely using **environment variables**.

## Licensing & Attribution
- Follow SoccerNet licensing terms; cite dataset and related papers.
- Do **not** redistribute raw videos or annotations.
- Share code and trained models only.

## Privacy & Data Minimization
- No re-identification of players, referees, or spectators.
- Store only minimal, task-relevant features; delete raw clips post-processing.

## Potential Risks
- **Bias:** Dataset may overrepresent certain leagues or broadcast conditions.
- **Privacy:** Visuals contain identifiable individuals (players/fans).
- **Security:** Mishandling API keys or raw data could expose sensitive information.

## Mitigations
- **Data handling:** Secure storage, restricted access, anonymization before sharing.
- **Bias checks:** Report data balance, evaluate generalization across competitions.
- **Security:** Environment-based key storage and compliance with all ToS limits.

## Limitations
- Broadcast-only data may not generalize to non-televised or youth-level matches.
- Performance may vary under unseen lighting, camera angles, or occlusions.
- Model results are **probabilistic**, not ground truth, and must be interpreted cautiously.

## Fairness & Transparency
- Report uncertainty intervals and dataset coverage.
- Publish a **Model Card** outlining data sources, limitations, metrics, and risks.
- Aligns with **ACM/IEEE** and **EU AI Act** principles of transparency, accountability, and fairness.

## Contact
- [Samyam Aryal](samyam.aryal@trojans.dsu.edu) for ethics, data removal, or opt-out requests.
