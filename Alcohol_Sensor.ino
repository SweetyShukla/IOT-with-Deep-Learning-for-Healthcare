const int AOUTpin=0;
int value;
int c=1;
float x,y;
void setup() {
  // put your setup code here, to run once:
  Serial.begin(19200);
  pinMode(AOUTpin,INPUT);
  delay(100);
}

void loop() {
  // put your main code here, to run repeatedly:
  value=analogRead(AOUTpin);
  x=((value*0.21)/1023);
  Serial.println(x);
  delay(5000);
}
