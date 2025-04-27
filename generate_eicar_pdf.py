from reportlab.pdfgen import canvas

eicar = (
    "X5O!P%@AP[4\\PZX54(P^)7CC)7)$EICAR-"
    "STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
)

c = canvas.Canvas("eicar.pdf")
c.setFont("Courier", 10)
c.drawString(72, 720, eicar)
c.save()
print("eicar.pdf created")
