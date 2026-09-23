"""Tests verifying the OpenJev container and deployment artifacts have no redundant local servers."""

from __future__ import annotations

from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy" / "diffusiongemma_jev"


def test_redundant_custom_server_and_entrypoint_are_removed() -> None:
    """deploy/diffusiongemma_jev must use razorback16/openjev rather than a hand-rolled server.py."""
    assert not (DEPLOY_DIR / "server.py").exists(), (
        "deploy/diffusiongemma_jev/server.py is redundant with razorback16/openjev and must be deleted"
    )
    assert not (DEPLOY_DIR / "entrypoint.sh").exists(), (
        "deploy/diffusiongemma_jev/entrypoint.sh is redundant with openjev-entrypoint and must be deleted"
    )


def test_dockerfile_extends_openjev_image() -> None:
    """Dockerfile must package razorback16/openjev with Cloud Run L4 defaults."""
    dockerfile = (DEPLOY_DIR / "Dockerfile").read_text(encoding="utf-8")
    assert "razorback16/openjev" in dockerfile
    assert "OPENJEV_PORT=8080" in dockerfile


def test_deploy_vm_script_and_single_engine_systemd() -> None:
    """deploy_vm.sh must deploy OpenJev to a GCE GPU VM and systemd must not run a duplicate vLLM engine."""
    deploy_vm = DEPLOY_DIR / "deploy_vm.sh"
    assert deploy_vm.exists(), "deploy/diffusiongemma_jev/deploy_vm.sh must exist"
    content = deploy_vm.read_text(encoding="utf-8")
    assert "gcloud compute" in content
    assert "razorback16/openjev:0.4.0" in content
    assert "--ipc=host" in content
    assert "a2-highgpu-1g" in content

    # OpenJev v0.3.0 embeds AsyncLLM directly in-process; a separate djev-vllm.service would double-load weights
    assert not (DEPLOY_DIR / "systemd" / "djev-vllm.service").exists(), (
        "djev-vllm.service is redundant because python -m openjev runs AsyncLLM in-process"
    )
    jev_service = (DEPLOY_DIR / "systemd" / "djev-jev.service").read_text(encoding="utf-8")
    assert "python -m openjev" in jev_service
    assert "Requires=djev-vllm.service" not in jev_service

