import serial.tools.list_ports

# List all available serial ports
ports = serial.tools.list_ports.comports()

for port in ports:
    print(f"Device: {port.device}")
    print(f"Description: {port.description}")
    print(f"Manufacturer: {port.manufacturer}\n")

# Simple logic to find the first "Arduino" device
arduino_port = next((p.device for p in ports if 'Arduino' in p.description), None)
print(f"Detected Arduino Port: {arduino_port}")