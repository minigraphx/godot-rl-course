# Course Download Packaging Design

**Date:** 2026-06-06

## Goal

Learners who read the course on the web must be able to download all files
needed for local work without cloning or navigating GitHub. They will still
install and run Godot and Python locally.

The course will provide both:

- per-unit starter and completed reference packages; and
- one complete-course package for offline or one-time download.

Third-party projects, plugins, and examples will remain separate and link to
their official sources.

## Packaging Model

Use a hybrid model generated from one source-controlled manifest:

1. A starter ZIP for each practical unit.
2. A solution ZIP for each practical unit.
3. A complete ZIP containing all unit starter and solution files.
4. A machine-readable artifact manifest with versions, sizes, checksums, and
   external dependencies.

This provides convenient web-course downloads without maintaining a second,
manually assembled distribution.

## Source Layout

Course-owned downloadable files will use a predictable structure:

```text
course-files/
  shared/
  unit-00/
    starter/
    solution/
  unit-neural-01/
    starter/
    solution/
```

Existing learner-facing examples will be migrated into this structure as each
unit is added to the download catalog. The manifest, not directory discovery,
is the authoritative package definition. This prevents unintended files such
as caches, logs, checkpoints, or local editor state from entering an archive.

Starter packages contain the minimum files needed to begin the unit. Solution
packages contain the completed reference implementation for that same unit.
Shared course-owned files are included in each package that declares them,
making every unit download usable on its own.

Generated archives and metadata are build artifacts and must not be committed
to the repository.

## Download Manifest

Add `course-downloads.yml` at the repository root. Each unit entry defines:

- stable unit ID;
- localized display names where needed;
- starter package name and source paths;
- solution package name and source paths;
- shared course-owned files;
- expected archive paths used for CI verification;
- external dependencies with official URLs and short installation notes;
- platform limitations;
- whether the unit has downloadable files.

Units that are theory-only remain visible in the course but do not require
empty ZIP files.

The manifest must use repository-relative source paths. Parent traversal,
absolute paths, symlinks escaping the repository, duplicate archive
destinations, and generated or ignored files are invalid.

## Generated Artifacts

The packaging script produces a staging tree equivalent to:

```text
downloads/
  latest/
    units/
      unit-00-starter.zip
      unit-00-solution.zip
    godot-rl-course-complete.zip
    manifest.json
    SHA256SUMS
  releases/
    v2026.06/
      units/
      godot-rl-course-complete.zip
      manifest.json
      SHA256SUMS
```

Archive filenames are stable and lowercase. Internal archive paths are also
stable so course instructions can tell learners exactly which project or file
to open.

`manifest.json` records:

- course version;
- build commit;
- artifact filename and relative URL;
- artifact byte size;
- SHA-256 checksum;
- unit and package type;
- external dependency metadata.

The complete package contains a top-level README explaining the starter and
solution layout, local prerequisites, course version, and external dependency
policy.

## Learner Experience

Each practical unit includes a "Files for this unit" panel near the top with:

- a starter download button;
- a completed reference solution button;
- version and approximate download size;
- required external downloads from official sources;
- a concise instruction for extracting and opening the project or script.

The solution is labeled as a reference implementation so learners do not
mistake it for the starting point.

A central Downloads page provides:

- the latest complete-course ZIP;
- links to immutable tagged releases;
- all per-unit starter and solution ZIPs;
- compatibility information matching the course release;
- checksums and basic integrity instructions;
- an explanation that Godot and Python still run locally.

English and German pages use the same artifacts. Labels and instructions may
be translated, but ZIP contents are not duplicated by locale unless a future
unit genuinely requires localized files.

## CI and Release Flow

Add a packaging workflow with separate validation, build, and publication
behavior.

### Pull Requests

On changes to the manifest, packaging code, course files, or download-facing
content, CI will:

1. validate the manifest schema and paths;
2. build every declared per-unit package;
3. build the complete package;
4. generate metadata and checksums;
5. reopen each ZIP and verify expected files;
6. reject empty packages, duplicate paths, and unsafe entries;
7. verify that generated download links resolve within the staged site; and
8. run `mkdocs build --strict`.

Pull requests do not publish downloads.

### Main Branch

Every successful merge to `main` rebuilds and publishes the moving `latest`
artifact set alongside the MkDocs site. `latest` records the source commit so
problems can be reproduced.

### Tagged Releases

A course release tag builds an immutable versioned artifact set. The workflow:

1. verifies that the tag's version matches the declared course version;
2. builds all artifacts from the tagged commit;
3. publishes them under a versioned site path;
4. creates or updates the corresponding GitHub Release; and
5. attaches the complete ZIP, `manifest.json`, and `SHA256SUMS` to that
   release.

Versioned site paths and release assets must never be overwritten by a later
build. Only `latest` is mutable.

## Hosting Portability

All generated links use site-relative URLs. The MkDocs output therefore works
on GitHub Pages and can later be copied unchanged to the owner's website.

The build output is a static directory and requires no application server.
When moving to another host, deployment only needs to preserve paths and serve
ZIP, JSON, and checksum files with normal static-file behavior.

GitHub Releases remain a secondary durable source for tagged complete-course
packages. The learner-facing pages link directly to the website-hosted files,
so no GitHub account or GitHub navigation is required.

## Error Handling

The build fails with a unit-specific message when:

- a declared source file is missing;
- a package has no files;
- two sources map to the same archive path;
- a source escapes the repository;
- an archive contains an unsafe path;
- expected files are absent after reopening the ZIP;
- metadata and actual size or checksum differ; or
- a generated download link is missing from the staged site.

Publication occurs only after all validation and the strict MkDocs build pass.
A failed release leaves the previously published `latest` and tagged
artifacts unchanged.

## Testing

The implementation must include focused automated tests for:

- manifest parsing and schema errors;
- package selection and destination mapping;
- rejected traversal and duplicate paths;
- deterministic archive contents;
- checksum and size generation;
- complete-bundle composition;
- generated download-page data; and
- tagged versus `latest` output paths.

CI also performs an integration test that builds the real manifest, inspects
every ZIP, and builds the full multilingual MkDocs site.

Deterministic ZIP ordering and normalized timestamps should be used so the same
source commit produces identical checksums across repeated CI builds.

## Scope Boundaries

This project includes course-owned file packaging, download UI, CI validation,
and static publication.

It does not:

- run Godot or Python in the browser;
- bundle Godot, Python, Conda, third-party plugins, or third-party examples;
- mirror upstream repositories;
- create user accounts or authenticated downloads;
- store learner progress; or
- add a custom backend service.

## Rollout

Implementation should proceed in three stages:

1. Build and validate the manifest-driven packaging pipeline using the
   existing Neural Foundations files as the first real package set.
2. Add the Downloads page and per-unit panels, then publish `latest` through
   the existing GitHub Pages deployment.
3. Inventory the remaining practical units and add starter and solution files
   incrementally, publishing the first immutable tagged course release only
   when the declared package coverage is complete.

This staged rollout avoids presenting empty or misleading download buttons
while allowing the packaging system to be tested on existing course-owned
material.
