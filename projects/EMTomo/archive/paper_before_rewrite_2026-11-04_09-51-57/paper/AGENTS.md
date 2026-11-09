# Updating the article PDF

The source is `paper/main.tex` with bibliography `paper/references.bib`. The service `paperpdf` in `../docker-compose.yml` is **nginx only**: it serves `paper/main.pdf` at port 8765; it does not compile LaTeX. The host may have no TeX installation.

From the `projects/EMTomo` directory, build using a one-off TeX Live Docker container (the first pull is large and may take a while):

```sh
docker pull texlive/texlive:latest
docker run --rm --user 1002:100 -v /mnt/disk01/egor/projects/EMTomo/paper:/work -w /work texlive/texlive:latest latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

`latexmk` runs the necessary LaTeX/Biber passes for `biblatex`; check that it exits successfully and `paper/main.pdf` has a recent modification time. `1002:100` is the user/group on this host (verify with `id -u; id -g` if the environment changes). The PDF is a tracked file; leave unrelated edits intact.

After rebuilding, recreate **only** the nginx PDF service to ensure its single-file bind mount sees the new PDF even if `latexmk` replaced the file inode:

```sh
docker compose up -d --no-deps --force-recreate paperpdf
```

The PDF is then available from the configured host at `http://10.0.62.59:8765/main.pdf`. Check `docker compose ps paperpdf`; the nginx config sets `Cache-Control: no-store`. Do not rebuild or restart `tomoviewer` just to update the paper.
