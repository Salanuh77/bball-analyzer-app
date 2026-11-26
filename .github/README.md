# CI and build delivery

`workflows/ci-cd.yml` runs on pushes and pull requests to `main`, and can also be started manually from GitHub Actions.

- Frontend: installs the lockfile dependencies, runs Expo ESLint and TypeScript.
- Backend: checks Python syntax and undefined-name/control-flow errors with Ruff, then compiles the Python modules.
- Delivery: after both checks pass, exports the Expo web app and uploads `dist/` as a downloadable Actions artifact, retained for 14 days. This does not publish a website or build native binaries.
- PR review: creates one summary comment on same-repository PRs and updates it on subsequent runs, with results and a link to diagnostics and artifacts. Fork PRs receive checks without a write-enabled comment job. These are automated static checks, not an AI or human review.

No extra secrets are needed. The comment job uses the built-in `GITHUB_TOKEN` with `pull-requests: write`; all jobs executing repository code have read-only permissions.

Run the same checks locally:

```sh
npm ci
npm run lint
npm run typecheck
npm run build:web
python -m pip install ruff==0.17.5
ruff check backend --select E9,F63,F7,F82
python -m compileall -q backend
```

The full `backend/test.py` video test is separate: it needs a running local backend, the sample video and trained model weights. CI does not download or execute those models.


# TO BE DONE:
 - Frontend UI for IOS & Android in React Native
 - Store user data in cloud database (AWS Terraform)
