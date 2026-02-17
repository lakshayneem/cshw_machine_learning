# Security Assessment Report

**Target:** http://testphp.vulnweb.com  
**Date:** 2026-02-17 15:40:34 IST  
**Tool:** RLM VAPT (Recursive Language Model Engine)

## 1. Executive Summary

The vulnerability assessment and penetration test (VAPT) conducted on http://testphp.vulnweb.com resulted in an overall security score of 25 out of 100, corresponding to a grade of E. This score reflects a security posture with significant weaknesses that require immediate attention to reduce the risk of exploitation. While no critical vulnerabilities were identified, the presence of multiple medium and high-severity findings indicates that the application’s current defenses are insufficient against common attack vectors. The volume of low-severity issues further suggests a lack of rigorous security hygiene and potential exposure to information leakage.

Key risks identified include a high-severity potential SQL injection vulnerability in the artists.php endpoint, which poses a direct threat to data integrity and confidentiality if exploited. Additionally, medium-severity issues such as potential cross-site scripting (XSS) via comment.php and multiple instances of information disclosure through site warning messages increase the attack surface and could facilitate further exploitation. These findings underscore the need for immediate remediation efforts focused on input validation, output encoding, and secure error handling to strengthen the application’s resilience against targeted attacks. Addressing these vulnerabilities will be critical to improving the overall security posture and protecting sensitive data.

## 2. Scope of Assessment

**In Scope:**
- http://testphp.vulnweb.com

**Out of Scope:**
- Denial of Service (DoS)
- Social Engineering
- Physical Security

## 3. Methodology

This assessment was conducted using an **AI-assisted VAPT approach**. The engine (RLM VAPT) utilized an LLM planner to strategically navigate the application, identify attack surfaces, and execute safe, read-only probes.

**Techniques Used:**
- Passive Reconnaissance (Header Analysis, Technology Detection)
- Active Probing (HTTP Method Enumeration, CORS Checks, Injection Probing)
- LLM-driven Surface Discovery

## 4. Vulnerability Summary

| ID | Vulnerability Name | Severity | CVSS |
|---|---|---|---|
| http_probe-_-1771323004 | Missing Security Headers | low | 3.5 |
| http_probe-_-1771323004-server | Server Version Disclosure | low | 3.3 |
| http_probe-_-1771323004-xpb | Technology Disclosure via X-Powered-By | low | 3.3 |
| http_probe-_-1771323004-llm-3 | Information Disclosure - Site Purpose Warning | medium | 0.0 |
| http_probe-_-1771323004-llm-4 | Potentially Outdated Technology - Flash Object | low | 0.0 |
| http_probe-_-1771323004-llm-5 | Potential Sensitive Information - Email Address Exposure | low | 0.0 |
| header_analysis-_index.php-1771323010 | Missing Security Headers | low | 3.5 |
| header_analysis-_index.php-1771323010-server | Server Version Disclosure | low | 3.3 |
| header_analysis-_index.php-1771323010-xpb | Technology Disclosure via X-Powered-By | low | 3.3 |
| header_analysis-_index.php-1771323010-llm-9 | Information Disclosure - Site Warning Message | medium | 0.0 |
| header_analysis-_index.php-1771323010-llm-10 | Potentially Outdated Flash Object | low | 0.0 |
| header_analysis-_index.php-1771323010-llm-11 | Search Form with GET Parameter in Action URL | low | 0.0 |
| http_probe-_categories.php-1771323018 | Missing Security Headers | low | 3.5 |
| http_probe-_categories.php-1771323018-server | Server Version Disclosure | low | 3.3 |
| http_probe-_categories.php-1771323018-xpb | Technology Disclosure via X-Powered-By | low | 3.3 |
| http_probe-_categories.php-1771323018-llm-15 | Information Disclosure - Warning Message | medium | 0.0 |
| http_probe-_categories.php-1771323018-llm-16 | Potentially Outdated JavaScript Code | low | 0.0 |
| http_probe-_categories.php-1771323018-llm-17 | Potential Sensitive Information in Comments | low | 0.0 |
| http_probe-_artists.php-1771323025 | Missing Security Headers | low | 3.5 |
| http_probe-_artists.php-1771323025-server | Server Version Disclosure | low | 3.3 |
| http_probe-_artists.php-1771323025-xpb | Technology Disclosure via X-Powered-By | low | 3.3 |
| http_probe-_artists.php-1771323025-llm-21 | Potential SQL Injection in artists.php | high | 0.0 |
| http_probe-_artists.php-1771323025-llm-22 | Potential Cross-Site Scripting (XSS) via comment.php | medium | 0.0 |
| http_probe-_artists.php-1771323025-llm-23 | Sensitive Information Disclosure in Warning Message | low | 0.0 |
| http_probe-_artists.php-1771323025-llm-24 | Use of Deprecated HTML and JavaScript | low | 0.0 |

## 5. Detailed Findings

### 5.1. [LOW] Missing Security Headers

**CVSS Score:** 3.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

Absent: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy

**Impact:**

Reduced protection against common web attacks like XSS, clickjacking, and mime-sniffing.

**Recommendation:**

Implement missing security headers: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy.

---

### 5.2. [LOW] Server Version Disclosure

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

Server header: nginx/1.19.0

**Impact:**

Attackers can use specific version information to find and exploit known vulnerabilities in the web server.

**Recommendation:**

Configure the server to hide version information (e.g., 'server_tokens off;' in nginx).

---

### 5.3. [LOW] Technology Disclosure via X-Powered-By

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

PHP/5.6.40-38+ubuntu20.04.1+deb.sury.org+1

**Impact:**

Discloses underlying technologies/frameworks, allowing attackers to tailor their exploits.

**Recommendation:**

Remove the X-Powered-By header from server responses.

---

### 5.4. [MEDIUM] Information Disclosure - Site Purpose Warning

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

The page contains a visible warning that this is an intentionally vulnerable test site designed to help test web vulnerability scanners and manual hacking skills. This indicates the site is not secure and is meant for testing, which could attract attackers.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.5. [LOW] Potentially Outdated Technology - Flash Object

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

The page embeds a Flash object (Flash/add.swf), which is deprecated and unsupported in modern browsers, potentially leading to security risks if exploited.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.6. [LOW] Potential Sensitive Information - Email Address Exposure

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/`

**Description & Evidence:**

The contact email wvs@acunetix.com is exposed in the page footer, which could be harvested for spam or phishing.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.7. [LOW] Missing Security Headers

**CVSS Score:** 3.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

Absent: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy

**Impact:**

Reduced protection against common web attacks like XSS, clickjacking, and mime-sniffing.

**Recommendation:**

Implement missing security headers: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy.

---

### 5.8. [LOW] Server Version Disclosure

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

Server header: nginx/1.19.0

**Impact:**

Attackers can use specific version information to find and exploit known vulnerabilities in the web server.

**Recommendation:**

Configure the server to hide version information (e.g., 'server_tokens off;' in nginx).

---

### 5.9. [LOW] Technology Disclosure via X-Powered-By

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

PHP/5.6.40-38+ubuntu20.04.1+deb.sury.org+1

**Impact:**

Discloses underlying technologies/frameworks, allowing attackers to tailor their exploits.

**Recommendation:**

Remove the X-Powered-By header from server responses.

---

### 5.10. [MEDIUM] Information Disclosure - Site Warning Message

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

The page contains a visible warning message indicating that this is an intentionally vulnerable test site. This may disclose to attackers that the site is a test environment and may contain vulnerabilities such as SQL Injection, XSS, and CSRF.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.11. [LOW] Potentially Outdated Flash Object

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

The page embeds a Flash object (Flash/add.swf) which is deprecated technology and may expose the site to security risks if the Flash content is vulnerable.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.12. [LOW] Search Form with GET Parameter in Action URL

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/index.php`

**Description & Evidence:**

The search form posts to 'search.php?test=query' which includes a GET parameter in the action URL. This could lead to unexpected behavior or parameter pollution if not handled properly.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.13. [LOW] Missing Security Headers

**CVSS Score:** 3.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

Absent: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy

**Impact:**

Reduced protection against common web attacks like XSS, clickjacking, and mime-sniffing.

**Recommendation:**

Implement missing security headers: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy.

---

### 5.14. [LOW] Server Version Disclosure

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

Server header: nginx/1.19.0

**Impact:**

Attackers can use specific version information to find and exploit known vulnerabilities in the web server.

**Recommendation:**

Configure the server to hide version information (e.g., 'server_tokens off;' in nginx).

---

### 5.15. [LOW] Technology Disclosure via X-Powered-By

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

PHP/5.6.40-38+ubuntu20.04.1+deb.sury.org+1

**Impact:**

Discloses underlying technologies/frameworks, allowing attackers to tailor their exploits.

**Recommendation:**

Remove the X-Powered-By header from server responses.

---

### 5.16. [MEDIUM] Information Disclosure - Warning Message

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

The page contains a visible warning message disclosing that this is an intentionally vulnerable test application. This may aid attackers by confirming the presence of vulnerabilities such as SQL Injection, XSS, and CSRF.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.17. [LOW] Potentially Outdated JavaScript Code

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

The page uses a JavaScript function MM_reloadPage targeting Netscape 4 browser, which is obsolete. While not a direct security issue, it indicates legacy code that may not be maintained or secured.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.18. [LOW] Potential Sensitive Information in Comments

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/categories.php`

**Description & Evidence:**

HTML comments reveal template file names and editing regions (e.g., /Templates/main_dynamic_template.dwt.php). This may provide attackers with insights into the server-side structure.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.19. [LOW] Missing Security Headers

**CVSS Score:** 3.5 (CVSS:3.1/AV:N/AC:L/PR:N/UI:R/S:U/C:N/I:L/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

Absent: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy

**Impact:**

Reduced protection against common web attacks like XSS, clickjacking, and mime-sniffing.

**Recommendation:**

Implement missing security headers: Strict-Transport-Security, X-Content-Type-Options, X-Frame-Options, Content-Security-Policy, Referrer-Policy, Permissions-Policy.

---

### 5.20. [LOW] Server Version Disclosure

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

Server header: nginx/1.19.0

**Impact:**

Attackers can use specific version information to find and exploit known vulnerabilities in the web server.

**Recommendation:**

Configure the server to hide version information (e.g., 'server_tokens off;' in nginx).

---

### 5.21. [LOW] Technology Disclosure via X-Powered-By

**CVSS Score:** 3.3 (CVSS:3.1/AV:N/AC:L/PR:N/UI:N/S:U/C:L/I:N/A:N)

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

PHP/5.6.40-38+ubuntu20.04.1+deb.sury.org+1

**Impact:**

Discloses underlying technologies/frameworks, allowing attackers to tailor their exploits.

**Recommendation:**

Remove the X-Powered-By header from server responses.

---

### 5.22. [HIGH] Potential SQL Injection in artists.php

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

The URL parameter 'artist' in 'artists.php?artist=1' is directly used in links without sanitization, indicating possible SQL Injection vulnerability.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.23. [MEDIUM] Potential Cross-Site Scripting (XSS) via comment.php

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

The 'comment.php' page is opened via JavaScript with 'aid' parameter, which may be vulnerable to XSS if input is not properly sanitized.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.24. [LOW] Sensitive Information Disclosure in Warning Message

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

The page contains a warning that it is intentionally vulnerable and for testing purposes, which may disclose information about the environment and encourage attacks.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

### 5.25. [LOW] Use of Deprecated HTML and JavaScript

**Similarity to:** OWASP Top 10 (A01/A05)

**Surface:** `/artists.php`

**Description & Evidence:**

The page uses HTML 4.01 Transitional and outdated JavaScript for page reload, which may lead to compatibility and security issues.

**Recommendation:**

Review the configuration and apply best practices to mitigate this risk.

---

## 6. Positive Security Findings

The following security controls were observed to be functioning correctly:

- **Authentication:** No obvious bypasses were found on protected endpoints.
- **Data Leakage:** No common sensitive keys (API keys, secrets) were detected in response bodies.
- **Coverage:** The automated planner explored multiple surfaces without triggering critical errors.

## 7. Overall Security Score & Rating

**Score:** 25 / 100

**Grade:** E

🔴 **Poor** (Red)

## 8. Comparative Assessment Time

| Assessment Type | Approx. Time |
| ---------------- | ------------ |
| Traditional VAPT | 3-5 Days |
| AI-assisted VAPT | < 1 Hour |

## 9. Conclusion

In conclusion, the application at http://testphp.vulnweb.com demonstrates significant security weaknesses, reflected in its low score of 25/100 and an overall grade of E, indicating it is currently unfit for production deployment. The identified vulnerabilities pose substantial risks that could be exploited by attackers, potentially leading to data breaches, unauthorized access, and service disruption. Immediate remediation is required to address critical security flaws before considering the application for any production use. It is strongly recommended that a comprehensive vulnerability management plan be implemented, including patching, secure coding practices, and rigorous retesting to ensure the application meets acceptable security standards.

## 10. Appendix: Resource Usage

| Metric | Count |
|---|---|
| Prompt Tokens | 12186 |
| Completion Tokens | 1717 |
| **Total Tokens** | **13903** |

