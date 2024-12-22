import csv
import random
import string

def generate_domain(tld=None):
    if tld is None:
        tld = random.choice(['.com', '.org', '.net', '.edu'])
    name_length = random.randint(5, 15)
    name = ''.join(random.choices(string.ascii_lowercase + string.digits, k=name_length))
    return f"{name}{tld}"

# Load disposable domains
with open('disposable-email-domains/disposable_email_blocklist.conf', 'r') as f:
    disposable_domains = f.read().splitlines()

# List of valid email domains (including more .edu domains)
valid_domains = [
    'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com', 'aol.com', 
    'nyu.edu', 'harvard.edu', 'mit.edu', 'stanford.edu', 'berkeley.edu',
    'columbia.edu', 'yale.edu', 'princeton.edu', 'cornell.edu', 'upenn.edu',
    'caltech.edu', 'uchicago.edu', 'duke.edu', 'northwestern.edu', 'jhu.edu'
]

# Create a CSV file for training
with open('email_domains.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['domain', 'label'])  # Header
    
    # Write disposable domains with label 1
    for domain in disposable_domains:
        writer.writerow([domain, 1])
    
    # Write valid domains with label 0
    for domain in valid_domains:
        writer.writerow([domain, 0])
    
    # Generate additional synthetic valid domains
    for _ in range(5000):
        domain = generate_domain()
        writer.writerow([domain, 0])
    
    # Generate synthetic disposable-looking domains
    for _ in range(2000):
        domain = generate_domain(tld=random.choice(['.xyz', '.top', '.site', '.online']))
        writer.writerow([domain, 1])

print("Dataset created: email_domains.csv")