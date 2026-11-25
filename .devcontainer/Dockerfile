FROM mcr.microsoft.com/devcontainers/universal:2
# The universal image ships a Yarn APT source whose public key is missing,
# so apt-get update fails (NO_PUBKEY 62D54F0403F625) when the docker-in-docker
# feature installs. Add Yarn's public key first so the feature install can run.
RUN curl -fsSL https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add -
