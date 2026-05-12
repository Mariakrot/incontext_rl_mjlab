FROM nvcr.io/nvidia/pytorch:25.12-py3

ARG DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-euxo", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    git sudo libglfw3-dev libglew-dev libosmesa6-dev patchelf \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/0.8.5/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin:$PATH"
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=1000
RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then groupadd --gid "${USER_GID}" "${USERNAME}"; fi; \
    if ! id -u "${USERNAME}" >/dev/null 2>&1; then useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}"; fi; \
    usermod -aG sudo "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

WORKDIR /workspace
RUN mkdir -p /home/ubuntu/.cache && chown -R ${USERNAME}:${USERNAME} /home/ubuntu
RUN chown -R ${USERNAME}:${USERNAME} /usr/local/lib/python3.12/dist-packages /usr/local/bin
ENV PATH="/home/${USERNAME}/.local/bin:$PATH"
USER ${USERNAME}
CMD ["bash"]