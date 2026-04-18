// -------- Pin Definitions --------
#define LDR_PIN A13
#define IR_PIN A1
#define SOUND_PIN A2   // BACK TO ANALOG
#define GAS_PIN A0

#define TRIG_PIN 7
#define ECHO_PIN 6

#define RELAY_PIN 10

// -------- Thresholds --------
int IR_THRESHOLD    = 50;
int SOUND_THRESHOLD = 80;   // you WILL tune this
int GAS_THRESHOLD   = 400;

// -------- Setup --------
void setup() {
  Serial.begin(9600);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
}

// -------- Ultrasonic --------
long readUltrasonic() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);

  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  long duration = pulseIn(ECHO_PIN, HIGH, 30000);
  long distance = duration * 0.034 / 2;

  if (distance == 0) return 999;
  return distance;
}

// -------- Main Loop --------
void loop() {

  int ldrValue   = analogRead(LDR_PIN);
  int irValue    = analogRead(IR_PIN);
  int soundValue = analogRead(SOUND_PIN);
  int gasValue   = analogRead(GAS_PIN);

  long distance = readUltrasonic();

  bool risk = false;

  if (gasValue > GAS_THRESHOLD) risk = true;
  if (irValue > IR_THRESHOLD) risk = true;

  // SOUND (analog spike detection)
  if (soundValue > SOUND_THRESHOLD) risk = true;

  digitalWrite(RELAY_PIN, risk ? HIGH : LOW);

  Serial.print("LDR:");
  Serial.print(ldrValue);
  Serial.print(",");

  Serial.print("IR:");
  Serial.print(irValue);
  Serial.print(",");

  Serial.print("SOUND:");
  Serial.print(soundValue);
  Serial.print(",");

  Serial.print("GAS:");
  Serial.print(gasValue);
  Serial.print(",");

  Serial.print("DIST:");
  Serial.print(distance);
  Serial.print(",");

  Serial.print("RISK:");
  Serial.println(risk);

  delay(100);
}