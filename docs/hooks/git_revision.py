"""MkDocs hook: inject git commit hash as a <meta> tag into every page."""

import subprocess


def _git_short_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


_COMMIT = _git_short_hash()


def on_page_context(context, **kwargs):
    context["git_commit"] = _COMMIT
    return context


def on_post_page(output, **kwargs):
    meta = f'<meta name="git-commit" content="{_COMMIT}">'
    return output.replace("</head>", f"{meta}\n</head>", 1)
