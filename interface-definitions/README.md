# Centralized Protobuf Definitions for AD Central Services

## 1. Introduction

This repository serves as the single source of truth for all Protocol Buffer (`.proto`) interface definitions used by control system services.

The primary goals of this centralized approach are:

- Single Source of Truth: Ensure all services use the same, up-to-date interface definitions.
- Consistency: Maintain uniform style, naming conventions, and structure across all definitions.
- Interoperability: Guarantee that services can communicate reliably using compatible message formats.
- Ease of Use: Simplify the process for developers to find, use, and update interface definitions.
- Clear Versioning: Manage changes and compatibility effectively over time.

## 2. Core Principles

- Backward Compatibility Preferred: Strive to make changes backward compatible. Breaking changes should be rare, planned, and clearly communicated.
- Automation: Leverage CI/CD pipelines for validation, code generation, artifact publishing, and version management.
- Clear Ownership: The process for proposing and approving changes is well-defined (see Workflow section).
- Language Agnostic Definitions: `.proto` files define the contract independent of implementation languages.

## 3. Repository Structure

```
interface-definitions/
├── proto/
│   ├── controls/
│   │   ├── common/         # Common types used across services
│   │   │   └── v1/
│   │   │       ├── timestamp.proto
│   │   │       └── identifiers.proto
│   │   └── services/       # Hold protocols specific to a service
│   │       ├── service_a/
│   │       │   └── v1/
│   │       │       ├── service_a_api.proto
│   │       │       └── service_a_messages.proto
│   │       └── ...         # Other services/domains
│   └── google/             # Standard Google well-known types
├── build/                  # Build scripts and configuration
│   ├── java/
│   │   └── build.gradle or pom.xml # Config for Java artifact generation
│   └── rust/
│       └── Cargo.toml            # Config for Rust artifact generation
│       └── build.rs              # Build script
├── .github/workflows/      # CI/CD pipeline definitions (e.g., GitHub Actions)
│   └── ci.yaml
├── .gitignore
└── README.md               # This file
```

- `proto/`: Root for all `.proto` files.
- `proto/<domain>/<api_version>/`: Group by domain/service, then by API contract version (`v1`, `v2`, etc.).
- `build/`: Contains language-specific build configurations needed to generate code and packages from the `.proto` files.

## 4. Versioning Strategy

Two types of versioning are critical:

- API Contract Versioning (Directory Structure):

  - Use version directories (`v1`, `v2`, etc.) within `proto/<domain>/`.
  - MINOR/PATCH Changes (Backward Compatible): Modify `.proto` files within the _current_ version directory (e.g., `v1`). Examples: adding optional fields, adding RPC methods, adding comments.
  - MAJOR Changes (Breaking): Create a _new_ version directory (e.g., `v2`). Copy/modify `.proto` files there. Examples: removing/renaming fields, changing field types, renaming RPCs.

- Generated Artifact Versioning (Git Tags & SemVer):
  - Generated code artifacts (JARs, Crates) are versioned using Semantic Versioning (MAJOR.MINOR.PATCH).
  - Git tags (e.g., `v1.2.0`) mark commits corresponding to released artifacts.
  - PATCH (`x.y.Z`): Increment for backward-compatible fixes in generated code/build scripts or docs-only `.proto` changes.
  - MINOR (`x.Y.z`): Increment for backward-compatible additions to `.proto` files (new optional fields, methods).
  - MAJOR (`X.y.z`): Increment _only_ for breaking changes in `.proto` files (requiring a new API `vX` directory). The artifact MAJOR version should align with the API `vX` directory (e.g., `v1` protos -> `1.x.y` artifacts).
    - TODO: Can we get a workflow to validate this?

## 5. Workflow for Changes

1.  Branch: Create a feature branch from `main`.
2.  Change: Modify/add `.proto` files following versioning rules. Update `build/` configs if needed.
3.  Commit & Push: Commit changes with clear commit messages. Push branch.
4.  Pull Request (PR): Open PR against `main`.
5.  CI Checks:
    - Required: `.proto` validation (linting via `buf lint`, breaking change detection via `buf breaking`, basic compilation check) must pass to allow the merge to main.
    - Informational: Full language builds (Java, Rust) may run (TODO: we can also just run this on main, but could be good to know on PR). Failures here do not block merge but indicate potential issues for the release process (triggered on merge to main).
6.  Review: Not required, but a good idea to get validation. Code review focuses on correctness, conventions, and backward compatibility.
7.  Merge: Merge approved PRs with passing _proto validation_ checks into `main`.
8.  Tag for Release: Create a versioned Git tag (e.g., `v1.2.0`) on the `main` branch commit you want to release. This tag triggers the release pipeline.
    - TODO: Should we bundle a release? Maybe just the standard GitHub zip file?

## 6. Build and Release Process

This repository automatically builds and publishes language-specific artifacts via CI/CD (GitHub Action Workflows) based on Git tags.

- Decoupled Build Process:
  - Merging only requires valid `.proto` files.
  - Releasing (triggered by a git tag) requires all language builds (Java, Rust) to succeed.
- Release Pipeline (on git tag):
  1.  Checks out the tagged commit.
  2.  Generates code for all target languages using configurations in `build/`.
  3.  Builds and packages artifacts (Java JARs, Rust crates).
  4.  Publishes artifacts to their respective repositories (see section 8).
- Release Success/Failure:
  - If all language builds/publishes succeed, the tag represents a successful release.
  - If any language build/publish fails (e.g., Java build issue), the overall release pipeline fails. Artifacts for that tag are not published or considered stable. The team must fix the build issue (in a new commit) and create a new Git tag (e.g., `v1.2.1`) to attempt the release again. A failed language build blocks the _release_, not the merging of `.proto` changes.

## 7. Language-Specific Artifacts

Generated artifacts are published centrally by the CI/CD pipeline in this repository. Do not generate code from `.proto` files directly in downstream service repositories.

### 7.1. Java (Maven/Gradle)

TODO: I need help with this. These are generic instructions.

- Artifact: Standard Java JAR files containing generated Protobuf classes.
- Build Config: `build/java/build.gradle` or `build/java/pom.xml`.
- Publishing (CI):
  - Uses Gradle/Maven with Protobuf plugins.
  - Publishes JARs to GitHub Packages (as a Maven repository).
  - Requires `GITHUB_TOKEN` with `write:packages` scope in CI secrets.
  - GroupId: `[e.g., gov.fnal.controls.protobuf]`
  - ArtifactId: `[e.g., proto-<domain>-<apiversion>-java]` (e.g., `proto-servicea-v1-java`)
- Consuming (Client to the Service):
  1.  Configure Gradle (`build.gradle`) or Maven (`pom.xml` + `settings.xml`) to use the GitHub Packages Maven repository associated with the `fermi-ad` organization. Authenticate using a `GITHUB_TOKEN`.
  2.  Add the dependency using the published coordinates:
      ```groovy
      // Gradle Example
      implementation 'gov.fnal.controls.protobuf:proto-servicea-v1-java:1.2.0'
      ```
      ```xml
      <dependency>
          <groupId>gov.fnal.controls.protobuf</groupId>
          <artifactId>proto-servicea-v1-java</artifactId>
          <version>1.2.0</version>
      </dependency>
      ```

### 7.2. Rust (Cargo)

- Artifact: Standard Rust crates containing generated Protobuf structs and gRPC traits.
- Build Config: `build/rust/Cargo.toml`, `build/rust/build.rs`.
- Publishing (CI):
  - Note: GitHub Packages does not currently offer native support for hosting Cargo registries. Choose one of the following methods:
  - Public - `crates.io`:
    - Suitable if the interface definitions are not considered proprietary or sensitive.
    - CI runs `cargo publish`.
    - Requires a `crates.io` API token stored as a `CARGO_REGISTRY_TOKEN` secret in the CI environment.
    - Crate Name: `[e.g., protobuf-servicea-v1]`
  - Private on GitHub - Private Git Repository:
    - Suitable for private/internal interfaces.
    - CI checks out a separate, dedicated private Git repository (e.g., `fermi-ad/proto-rust-crates`).
    - CI copies the generated Rust crate source code into this repo, commits, tags (matching the proto repo tag, e.g., `v1.2.0`), and pushes.
    - Requires an SSH deploy key or token with write access to the dedicated crate repository configured as a CI secret.
  - Private on-prem - Private Registry, like Artifactory:
    - Configure CI to publish there. Requires registry-specific setup and credentials.
- Consuming (Client to the Service):
  - From `crates.io`: Add to `Cargo.toml`:
    ```toml
    [dependencies]
    protobuf-servicea-v1 = "1.2.0"
    ```
  - From Private Git Repository: Add to `Cargo.toml`:
    ```toml
    [dependencies]
    # Use the crate name defined in the published crate's Cargo.toml
    protobuf-servicea-v1 = { git = "ssh://git@github.com/fermi-ad/proto-rust-crates.git", tag = "v1.2.0" }
    ```

## 8. Best Practices for `.proto` Definitions

- Use clear, consistent naming: `UpperCamelCase` for messages/enums/services, `lower_snake_case` for fields.
- Add comments (`//`) explaining fields, messages, services, RPCs, and potential evolution.
- Prefer adding _new_ optional fields over changing existing ones. Use unique field numbers.
- Use `reserved` for field numbers/names of deleted fields to prevent accidental reuse.
- Use `deprecated=true` field option for fields planned for removal.
- Define services (`service`) and RPC methods (`rpc`) clearly.
- Use standard "Well-Known Types" like `google.protobuf.Timestamp` where appropriate.
- Validate locally using `buf lint` before pushing changes.

## 9. Contact

For questions or issues regarding this repository or the Protobuf workflow, please create and issue.
