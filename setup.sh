sudo apt update
sudo apt upgrade -y
sudo apt install openjdk-17-jdk -y
cd synthea
./gradlew build
./gradlew build check test
