Collaboration Controller (GitHub‑First)
=======================================

**Who is the controller?**  
GitHub is the *source of truth* and the collaboration bus. ChatGPT, Claude, and Copilot all work *through GitHub*:

- **ChatGPT**: prepares ready-to-merge files/patches, commit messages, CI config, docs.
- **Claude**: can pull the same repo/branch, reconcile external builds, and open PRs.
- **Copilot**: assists in-editor, suggests changes on the same repo, and comments in PRs.
- **You**: approve merges and run the provided push scripts when ChatGPT gives you a patch.

This pack gives you:
- A CI workflow that lints, sanity-checks the dashboard API, and publishes a build summary.
- PR & Issue templates that force evidence artifacts and clear acceptance criteria.
- CODEOWNERS so reviews are auto-requested.
- A push helper script to stage/push common paths quickly.
- A starter `ai_shared` heartbeat file for cross‑agent status.

---

Quick Start
-----------
1) Place these files in your repo root (`C:\ai_project_hub\`).
2) Commit & push:
   ```powershell
   cd C:\ai_project_hub
   git add .github CODEOWNERS README_COLLAB.md scripts store/ai_shared
   git commit -m "cont: add collaboration controller pack (CI, templates, CODEOWNERS, ai_shared)"
   git push
   ```
3) In GitHub → Settings → Branches → Protect `main`:
   - Require PRs, 1+ approval, and status checks pass (select the workflow added here).
   - Dismiss stale approvals on new commits.
   - Require linear history (optional).

4) Tell Claude/Copilot: “Work off branch `feat/ui-starter` and open PRs targeting `main`.”

---

Roles in practice
-----------------
- ChatGPT: Generates files, docs, and exact commands; you run `scripts/push_patch.ps1` to publish.
- Claude: Runs diffs against external builds, contributes patches via PR, writes reconciliation notes.
- Copilot: Helps author code in your editor; participates in PR reviews/comments.
- GitHub Actions: Runs lint/tests, posts a summary in the PR, and blocks merges on failures.

Artifacts
---------
- `store/ai_shared/mesh_heartbeat.json`: AIs write light heartbeat/status here.
- `store/code/IBKR_Algo_BOT/data/signals_tail.log`: Signals stream for runtime checks (not required in CI).