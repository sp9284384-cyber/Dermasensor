#include <DHT.h>

// Adjust these pins to match the ESP32 wiring.
#define DHT_PIN 4
#define DHT_TYPE DHT11
#define MOISTURE_PIN 34

// Calibrate these two values from your moisture probe.
const int MOISTURE_DRY_RAW = 4095;
const int MOISTURE_WET_RAW = 1200;

DHT dht(DHT_PIN, DHT_TYPE);

void setup() {
  Serial.begin(115200);
  dht.begin();
  analogReadResolution(12);
}

void loop() {
  float temperature = dht.readTemperature();
  int moistureRaw = analogRead(MOISTURE_PIN);

  if (isnan(temperature)) {
    delay(2000);
    return;
  }

  int moisture = map(
    moistureRaw,
    MOISTURE_DRY_RAW,
    MOISTURE_WET_RAW,
    0,
    100
  );
  moisture = constrain(moisture, 0, 100);

  Serial.print("{\"temperature\":");
  Serial.print(temperature, 1);
  Serial.print(",\"moisture\":");
  Serial.print(moisture);
  Serial.println("}");

  delay(2000);
}
