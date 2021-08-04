echo "Installing Docker"

echo "Uninstalling old versions"
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get update

echo "Installing dependencies"
sudo apt-get install apt-transport-https ca-certificates curl gnupg lsb-release -y

echo "Adding Docker gpg key"
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

echo "Setting up stable repository"
echo \
  "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "Installing Docker Engine"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y

echo "Confirm instillation"
sudo docker run --rm hello-world