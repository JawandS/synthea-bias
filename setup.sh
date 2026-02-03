sudo apt update
sudo apt upgrade -y
sudo apt install openjdk-17-jdk -y
git config --global user.name "Jawand"
git config --global user.email "JawandSingh@gmail.com"
cd synthea
./gradlew clean
./gradlew build
