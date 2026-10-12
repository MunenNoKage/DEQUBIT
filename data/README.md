# data/ — data card

Fill this in early, not at the end: settling a dataset's provenance and license before
building on it is much cheaper than discovering afterwards that it cannot be
redistributed, or that its origin cannot be stated.

## Name and source

TODO

## License and terms of use

TODO

## Size and format

TODO

## Schema

TODO

## How to get it

TODO

## Splits

TODO

## Personal data

TODO

## Scraping

TODO

## Known limitations and biases

TODO

## Ethical risks

TODO

## Version and checksum

TODO

---

## If the data is too large or private for this pattern

- **Large but shareable**: Git LFS, DVC + cloud storage, or Hugging Face Datasets.
- **Already hosted**: document the URL and the download command rather than re-hosting.
- **You need a DOI**: Zenodo.
- **Private**: keep it out of git entirely and document how to request access; graders
  and CI need a documented path too.

> [!warning]
> `git rm big_file` does **not** remove it from git history; it stays in every earlier
> commit, downloadable by anyone with repo access. If restricted data is ever committed
> by mistake, treat it as a leaked secret: rewrite history (`git filter-repo`) before
> anyone else pulls, or revoke access at the source. Choosing the storage approach
> before the first commit is far cheaper than fixing this afterwards.
