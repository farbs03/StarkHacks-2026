void setup() {
  Serial.begin(9600);
}

void loop() {
  int sound = analogRead(A2);
  Serial.println(sound);
  delay(50);
}