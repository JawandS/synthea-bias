sudo apt update
sudo apt upgrade -y
sudo apt install openjdk-17-jdk -y
git config user.name "Jawand"
git config user.email "JawandSingh@gmail.com"
cd synthea
./gradlew build
./gradlew build check test
