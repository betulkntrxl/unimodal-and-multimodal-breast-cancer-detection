# Research showcase website

Static source of the pink research showcase, with interactive results, publication links, original thesis, and self-hosted fonts.

## Preview

From the repository root run:

```sh
python3 -m http.server 8000 --directory website
```

Open http://localhost:8000. Serve website/ as the web root because assets use root-relative paths. No installation or build is needed.

Edit index.html for content and interactions, design.css for the visual design. Font licences are included in fonts/.

Hosted site: https://betul-breast-cancer-research.betulkantaroglu12.chatgpt.site/

The hosted Site currently requires owner access. This commit does not change Site sharing or enable GitHub Pages.
