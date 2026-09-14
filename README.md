# AI Learning Notes

A personal AI knowledge base built around understanding: from intuition, to mathematics, to implementation.

## Local development

```bash
npm install
npm run docs:dev
```

VitePress turns the Markdown files in `docs/` into a static website. The npm scripts above are defined in `package.json`.

## Useful commands

```bash
npm run docs:dev      # start the local writing server
npm run docs:audit    # verify local Markdown links and anchors
npm run docs:build    # build the production site
npm run docs:preview  # preview the production build
```

## Repository structure

```text
docs/
├── mathematics/       # mathematical foundations
├── deep-learning/     # neural networks and architectures
├── generative-models/ # latent variables, VAE and CVAE
├── llm/               # language-model notes
├── robot-learning/    # imitation learning, ACT, VLA
├── embodied-ai/       # embodied intelligence
├── projects/          # implementations and experiments
└── .vitepress/        # navigation, theme and site configuration
```

The GitHub repository is the source of truth. Notes, code and experiments should be changed here first; other platforms are distribution channels.

## Deployment

`.github/workflows/deploy.yml` builds and deploys the site with GitHub Pages on every push to `main`. In the repository settings, choose **GitHub Actions** as the Pages source.

The workflow automatically uses `/<repository-name>/` for a project Pages site. A custom domain can use `/` by setting the `BASE_PATH` environment variable to `/` in the workflow.
