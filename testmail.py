import smtplib

host = "localhost"
port = 1025  # Local SMTP server port
FROM = "testpython@test.com"
TO = "mukesh@adarone.com"
MSG = "TEST TEST!"

# Connect to the local SMTP server
server = smtplib.SMTP(host, port)
server.sendmail(FROM, TO, MSG)

server.quit()
print("Email Sent")
