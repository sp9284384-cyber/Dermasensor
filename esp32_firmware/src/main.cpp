#include <Arduino.h>
#include <DHT.h>

// Support both GPIO 4 (G4) and GPIO 0 (G0) automatically
DHT dht4(4, DHT11);
DHT dht0(0, DHT11);

void setup() {
  Serial.begin(115200);
  pinMode(4, INPUT_PULLUP);
  pinMode(0, INPUT_PULLUP);
  dht4.begin();
  dht0.begin();
}

void loop() {
  // 1. Try reading from GPIO 4 (G4)
  float temperature = dht4.readTemperature();
  float moisture = dht4.readHumidity();

  // 2. If Pin 4 failed, fallback to GPIO 0 (G0)
  if (isnan(temperature) || isnan(moisture)) {
    temperature = dht0.readTemperature();
    moisture = dht0.readHumidity();
  }

  // 3. If both failed, output helpful diagnostic error
  if (isnan(temperature) || isnan(moisture)) {
    Serial.println("{\"error\":\"DHT11 not responding on G4 or G0. Check VCC (3.3V), GND, and DATA wire\"}");
    delay(2000);
    return;
  }

  int moistureInt = constrain((int)round(moisture), 0, 100);

  Serial.print("{\"temperature\":");
  Serial.print(temperature, 1);
  Serial.print(",\"moisture\":");
  Serial.print(moistureInt);
  Serial.println("}");

  delay(2000);
}
