import sendgrid
from sendgrid.helpers.mail import *
import os


def send_email(name, role, requestor_email, request_focus):
    sg = sendgrid.SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("sundaiclub@gmail.com")
    to_email = To("sundaiclub@gmail.com")
    subject = "AI-NEWS-HOUND: New Visualization Request"
    email_text = f"""
            New Request Submitted:
            Name: {name}
            Role: {role}
            Email: {requestor_email}
            Research Focus Request: {request_focus}
            """
    content = Content("text/plain", email_text)
    mail = Mail(from_email, to_email, subject, content)
    mail.add_to(To("nader_k@mit.edu"))
    sg.client.mail.send.post(request_body=mail.get())
