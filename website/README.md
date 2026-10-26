# Research showcase website

Static source for the pink breast cancer research showcase, including Newsreader headings, Libre Franklin body text, interactive results, publication links, and the original thesis PDF.

## Preview locally

From the repository root:

```sh
python3 -m http.server 8000 --directory website
```

Open http://localhost:8000. Serve this folder as the web root; asset and PDF links use root-relative paths. No dependencies or build step are required.

## Deploy on Vercel

Import this GitHub repository and set **Root Directory** to `website`. Select **Other** as the framework. The included `vercel.json` declares static hosting with no installation or build step and uses this directory as output. Deploy from `main`.

## Files

- `index.html`: content and interactive results comparison
- `design.css`: pink palette, editorial typography, and responsive styles
- `fonts/editorial.css` and `fonts/editorial-*.ttf`: self-hosted Newsreader and Libre Franklin
- `fonts/*-OFL.txt`: font licences
- `papers/betul-kantaroglu-thesis.pdf`: original thesis

Original hosted site: https://betul-breast-cancer-research.betulkantaroglu12.chatgpt.site/
