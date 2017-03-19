# install terminator
sudo apt-get install terminator

# set up case insensitive autocomplete
if [ ! -a ~/.inputrc ]; then echo '$include /etc/inputrc' > ~/.inputrc; fi
echo 'set completion-ignore-case On' >> ~/.inputrc

# install git and gitg
sudo apt-get install git
sudo apt-get install gitg
