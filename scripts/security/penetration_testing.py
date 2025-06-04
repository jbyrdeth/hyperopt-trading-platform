#!/usr/bin/env python3
"""
Comprehensive Penetration Testing Framework for HyperOpt Platform
Automated security testing including vulnerability scanning, auth bypass, and API security.
"""

import asyncio
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import aiohttp
import requests
from urllib.parse import urljoin, urlparse
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SecurityTest:
    """Represents a single security test."""
    name: str
    category: str
    severity: str
    description: str
    test_function: str
    expected_result: str

@dataclass
class TestResult:
    """Represents the result of a security test."""
    test_name: str
    status: str  # PASS, FAIL, ERROR
    severity: str
    details: str
    evidence: Optional[str] = None
    timestamp: Optional[datetime] = None

class HyperOptPenetrationTester:
    """Main penetration testing framework for HyperOpt platform."""
    
    def __init__(self, config_path: str = "config/pentest_config.yaml"):
        """Initialize the penetration tester."""
        self.config = self._load_config(config_path)
        self.base_url = self.config['target']['base_url']
        self.api_url = self.config['target']['api_url']
        self.results: List[TestResult] = []
        self.session = requests.Session()
        self.setup_session()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file {config_path} not found, using defaults")
            return self._get_default_config()
        
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'target': {
                'base_url': 'https://api.hyperopt.company',
                'api_url': 'https://api.hyperopt.company/api/v1',
                'auth_url': 'https://auth.hyperopt.company',
                'app_url': 'https://app.hyperopt.company'
            },
            'authentication': {
                'test_username': 'pentest@example.com',
                'test_password': 'test_password_123',
                'admin_username': 'admin@hyperopt.company',
                'admin_password': 'admin_password_123'
            },
            'timeouts': {
                'request_timeout': 30,
                'scan_timeout': 300
            },
            'reporting': {
                'output_dir': '/tmp/pentest-reports',
                'formats': ['json', 'html', 'pdf']
            }
        }
    
    def setup_session(self):
        """Setup HTTP session with proper headers."""
        self.session.headers.update({
            'User-Agent': 'HyperOpt-PenTest/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        self.session.verify = True  # Enable SSL verification
        
    async def run_all_tests(self) -> List[TestResult]:
        """Run all penetration tests."""
        logger.info("Starting comprehensive penetration testing...")
        
        # Authentication Tests
        await self.test_authentication_bypass()
        await self.test_weak_passwords()
        await self.test_session_management()
        await self.test_oauth_vulnerabilities()
        
        # API Security Tests
        await self.test_api_authentication()
        await self.test_api_rate_limiting()
        await self.test_api_input_validation()
        await self.test_api_authorization()
        await self.test_sql_injection()
        await self.test_nosql_injection()
        
        # Web Application Tests
        await self.test_xss_vulnerabilities()
        await self.test_csrf_protection()
        await self.test_clickjacking_protection()
        await self.test_security_headers()
        
        # Infrastructure Tests
        await self.test_ssl_configuration()
        await self.test_server_information_disclosure()
        await self.test_directory_traversal()
        await self.test_file_upload_vulnerabilities()
        
        # Business Logic Tests
        await self.test_privilege_escalation()
        await self.test_data_exposure()
        await self.test_trading_logic_bypass()
        
        # Container Security Tests
        await self.test_container_escape()
        await self.test_kubernetes_misconfig()
        
        logger.info(f"Completed {len(self.results)} security tests")
        return self.results
    
    async def test_authentication_bypass(self):
        """Test for authentication bypass vulnerabilities."""
        test_name = "Authentication Bypass"
        logger.info(f"Running {test_name}...")
        
        try:
            # Test 1: Direct API access without authentication
            response = self.session.get(f"{self.api_url}/users/me")
            if response.status_code != 401:
                self._add_result(test_name, "FAIL", "HIGH", 
                               f"API allows unauthenticated access: {response.status_code}")
                return
            
            # Test 2: Authentication bypass with manipulated headers
            bypass_headers = {
                'X-User-ID': '1',
                'X-Admin': 'true',
                'X-Forwarded-User': 'admin',
                'Authorization': 'Bearer fake_token'
            }
            
            for header, value in bypass_headers.items():
                test_headers = self.session.headers.copy()
                test_headers[header] = value
                response = self.session.get(f"{self.api_url}/users/me", headers=test_headers)
                
                if response.status_code == 200:
                    self._add_result(test_name, "FAIL", "CRITICAL", 
                                   f"Authentication bypass via {header} header")
                    return
            
            # Test 3: SQL injection in login
            sql_payloads = [
                "admin' OR '1'='1",
                "admin' OR '1'='1' --",
                "admin' OR '1'='1' /*",
                "' UNION SELECT 1,1,1 --"
            ]
            
            for payload in sql_payloads:
                login_data = {
                    'username': payload,
                    'password': 'any_password'
                }
                response = self.session.post(f"{self.api_url}/auth/login", json=login_data)
                
                if response.status_code == 200:
                    self._add_result(test_name, "FAIL", "CRITICAL", 
                                   f"SQL injection in login: {payload}")
                    return
            
            self._add_result(test_name, "PASS", "INFO", "No authentication bypass vulnerabilities found")
            
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_weak_passwords(self):
        """Test for weak password policies."""
        test_name = "Weak Password Policy"
        logger.info(f"Running {test_name}...")
        
        try:
            weak_passwords = [
                "123456", "password", "admin", "hyperopt", "12345678",
                "qwerty", "abc123", "password123", "admin123"
            ]
            
            for password in weak_passwords:
                registration_data = {
                    'username': f'test_{password}@example.com',
                    'password': password,
                    'email': f'test_{password}@example.com'
                }
                
                response = self.session.post(f"{self.api_url}/auth/register", json=registration_data)
                
                if response.status_code == 201:
                    self._add_result(test_name, "FAIL", "MEDIUM", 
                                   f"Weak password accepted: {password}")
                    return
            
            self._add_result(test_name, "PASS", "INFO", "Strong password policy enforced")
            
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_api_rate_limiting(self):
        """Test API rate limiting implementation."""
        test_name = "API Rate Limiting"
        logger.info(f"Running {test_name}...")
        
        try:
            # Send rapid requests to trigger rate limiting
            endpoint = f"{self.api_url}/strategies"
            request_count = 100
            rate_limited = False
            
            for i in range(request_count):
                response = self.session.get(endpoint)
                
                if response.status_code == 429:  # Too Many Requests
                    rate_limited = True
                    break
                
                if i % 10 == 0:
                    logger.info(f"Sent {i} requests, status: {response.status_code}")
                
                await asyncio.sleep(0.1)  # Small delay between requests
            
            if not rate_limited:
                self._add_result(test_name, "FAIL", "MEDIUM", 
                               f"No rate limiting after {request_count} requests")
            else:
                self._add_result(test_name, "PASS", "INFO", "Rate limiting properly implemented")
                
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_sql_injection(self):
        """Test for SQL injection vulnerabilities."""
        test_name = "SQL Injection"
        logger.info(f"Running {test_name}...")
        
        try:
            # SQL injection payloads
            sql_payloads = [
                "' OR '1'='1",
                "' OR '1'='1' --",
                "' OR '1'='1' /*",
                "'; DROP TABLE users; --",
                "' UNION SELECT 1,2,3,4,5 --",
                "' AND (SELECT COUNT(*) FROM information_schema.tables) > 0 --",
                "' AND (SELECT SUBSTRING(@@version,1,1)) = '5' --"
            ]
            
            # Test endpoints that might be vulnerable
            test_endpoints = [
                "/strategies",
                "/optimizations",
                "/users",
                "/reports"
            ]
            
            for endpoint in test_endpoints:
                for payload in sql_payloads:
                    # Test in URL parameters
                    url = f"{self.api_url}{endpoint}?id={payload}"
                    response = self.session.get(url)
                    
                    # Check for SQL error messages
                    if self._check_sql_errors(response.text):
                        self._add_result(test_name, "FAIL", "CRITICAL", 
                                       f"SQL injection in {endpoint}: {payload}")
                        return
                    
                    # Test in POST data
                    post_data = {"id": payload, "search": payload}
                    response = self.session.post(f"{self.api_url}{endpoint}", json=post_data)
                    
                    if self._check_sql_errors(response.text):
                        self._add_result(test_name, "FAIL", "CRITICAL", 
                                       f"SQL injection in POST {endpoint}: {payload}")
                        return
            
            self._add_result(test_name, "PASS", "INFO", "No SQL injection vulnerabilities found")
            
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    def _check_sql_errors(self, response_text: str) -> bool:
        """Check response for SQL error messages."""
        sql_errors = [
            "SQL syntax error",
            "mysql_fetch_array",
            "ORA-01756",
            "Microsoft OLE DB Provider",
            "PostgreSQL query failed",
            "Warning: pg_exec",
            "valid MySQL result",
            "SQLServer JDBC Driver",
            "OLE/DB provider returned message",
            "SQLite error",
            "sqlite3.OperationalError"
        ]
        
        response_lower = response_text.lower()
        return any(error.lower() in response_lower for error in sql_errors)
    
    async def test_xss_vulnerabilities(self):
        """Test for Cross-Site Scripting (XSS) vulnerabilities."""
        test_name = "XSS Vulnerabilities"
        logger.info(f"Running {test_name}...")
        
        try:
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "javascript:alert('XSS')",
                "<iframe src='javascript:alert(\"XSS\")'></iframe>",
                "';alert('XSS');//",
                "\"><script>alert('XSS')</script>",
                "<body onload=alert('XSS')>"
            ]
            
            # Test endpoints that might reflect user input
            test_endpoints = [
                "/search",
                "/strategies/create",
                "/users/profile",
                "/reports/generate"
            ]
            
            for endpoint in test_endpoints:
                for payload in xss_payloads:
                    # Test in URL parameters
                    url = f"{self.api_url}{endpoint}?q={payload}"
                    response = self.session.get(url)
                    
                    if payload in response.text and 'script' in response.text:
                        self._add_result(test_name, "FAIL", "HIGH", 
                                       f"Reflected XSS in {endpoint}: {payload}")
                        return
                    
                    # Test in POST data
                    post_data = {"name": payload, "description": payload}
                    response = self.session.post(f"{self.api_url}{endpoint}", json=post_data)
                    
                    if payload in response.text and 'script' in response.text:
                        self._add_result(test_name, "FAIL", "HIGH", 
                                       f"Stored XSS in POST {endpoint}: {payload}")
                        return
            
            self._add_result(test_name, "PASS", "INFO", "No XSS vulnerabilities found")
            
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_security_headers(self):
        """Test for proper security headers."""
        test_name = "Security Headers"
        logger.info(f"Running {test_name}...")
        
        try:
            response = self.session.get(self.base_url)
            headers = response.headers
            
            # Required security headers
            required_headers = {
                'Strict-Transport-Security': 'HSTS header missing',
                'X-Frame-Options': 'Clickjacking protection missing',
                'X-Content-Type-Options': 'MIME-type sniffing protection missing',
                'X-XSS-Protection': 'XSS protection header missing',
                'Content-Security-Policy': 'CSP header missing',
                'Referrer-Policy': 'Referrer policy missing'
            }
            
            missing_headers = []
            for header, description in required_headers.items():
                if header not in headers:
                    missing_headers.append(f"{header}: {description}")
            
            if missing_headers:
                self._add_result(test_name, "FAIL", "MEDIUM", 
                               f"Missing security headers: {', '.join(missing_headers)}")
            else:
                self._add_result(test_name, "PASS", "INFO", "All security headers present")
                
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_ssl_configuration(self):
        """Test SSL/TLS configuration."""
        test_name = "SSL Configuration"
        logger.info(f"Running {test_name}...")
        
        try:
            # Use external tool to test SSL configuration
            domain = urlparse(self.base_url).netloc
            
            # Run testssl.sh if available
            try:
                result = subprocess.run(
                    ['testssl.sh', '--quiet', '--jsonfile', '/tmp/ssl_test.json', domain],
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                
                if result.returncode == 0 and Path('/tmp/ssl_test.json').exists():
                    with open('/tmp/ssl_test.json', 'r') as f:
                        ssl_results = json.load(f)
                    
                    # Check for SSL vulnerabilities
                    vulnerabilities = []
                    for test in ssl_results.get('scanResult', []):
                        if test.get('severity') in ['HIGH', 'CRITICAL']:
                            vulnerabilities.append(test.get('finding', 'Unknown SSL issue'))
                    
                    if vulnerabilities:
                        self._add_result(test_name, "FAIL", "HIGH", 
                                       f"SSL vulnerabilities: {', '.join(vulnerabilities)}")
                    else:
                        self._add_result(test_name, "PASS", "INFO", "SSL configuration secure")
                else:
                    # Fallback to basic SSL test
                    response = self.session.get(self.base_url)
                    if response.url.startswith('https://'):
                        self._add_result(test_name, "PASS", "INFO", "HTTPS properly configured")
                    else:
                        self._add_result(test_name, "FAIL", "HIGH", "HTTPS not enforced")
                        
            except subprocess.TimeoutExpired:
                self._add_result(test_name, "ERROR", "MEDIUM", "SSL test timeout")
            except FileNotFoundError:
                # testssl.sh not available, do basic check
                response = self.session.get(self.base_url)
                if response.url.startswith('https://'):
                    self._add_result(test_name, "PASS", "INFO", "Basic HTTPS check passed")
                else:
                    self._add_result(test_name, "FAIL", "HIGH", "HTTPS not enforced")
                    
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    async def test_trading_logic_bypass(self):
        """Test for business logic bypasses in trading functionality."""
        test_name = "Trading Logic Bypass"
        logger.info(f"Running {test_name}...")
        
        try:
            # Test negative amounts
            bypass_tests = [
                {"amount": -1000, "description": "negative investment amount"},
                {"leverage": 1000000, "description": "excessive leverage"},
                {"stop_loss": -100, "description": "negative stop loss"},
                {"take_profit": -100, "description": "negative take profit"}
            ]
            
            for test_case in bypass_tests:
                trade_data = {
                    "strategy_id": 1,
                    "amount": test_case.get("amount", 1000),
                    "leverage": test_case.get("leverage", 1),
                    "stop_loss": test_case.get("stop_loss", 0.05),
                    "take_profit": test_case.get("take_profit", 0.1)
                }
                
                response = self.session.post(f"{self.api_url}/trades", json=trade_data)
                
                if response.status_code == 201:
                    self._add_result(test_name, "FAIL", "HIGH", 
                                   f"Business logic bypass: {test_case['description']}")
                    return
            
            self._add_result(test_name, "PASS", "INFO", "Trading logic properly validated")
            
        except Exception as e:
            self._add_result(test_name, "ERROR", "MEDIUM", f"Test error: {str(e)}")
    
    def _add_result(self, test_name: str, status: str, severity: str, details: str):
        """Add a test result."""
        result = TestResult(
            test_name=test_name,
            status=status,
            severity=severity,
            details=details,
            timestamp=datetime.now()
        )
        self.results.append(result)
        
        # Log the result
        log_level = logging.ERROR if status == "FAIL" else logging.INFO
        logger.log(log_level, f"{test_name}: {status} - {details}")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive penetration testing report."""
        summary = {
            "total_tests": len(self.results),
            "passed": len([r for r in self.results if r.status == "PASS"]),
            "failed": len([r for r in self.results if r.status == "FAIL"]),
            "errors": len([r for r in self.results if r.status == "ERROR"])
        }
        
        # Group by severity
        severity_counts = {}
        for result in self.results:
            if result.status == "FAIL":
                severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "target": self.base_url,
            "summary": summary,
            "severity_breakdown": severity_counts,
            "results": [
                {
                    "test_name": r.test_name,
                    "status": r.status,
                    "severity": r.severity,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None
                }
                for r in self.results
            ]
        }
        
        return report
    
    def save_report(self, output_path: str = "/tmp/pentest-reports/hyperopt_pentest_report.json"):
        """Save the penetration testing report."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        report = self.generate_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved to {output_path}")
        
        # Also create a summary
        summary_path = output_path.replace('.json', '_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("HyperOpt Platform Penetration Testing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {report['timestamp']}\n")
            f.write(f"Target: {report['target']}\n\n")
            f.write("Summary:\n")
            f.write(f"  Total Tests: {report['summary']['total_tests']}\n")
            f.write(f"  Passed: {report['summary']['passed']}\n")
            f.write(f"  Failed: {report['summary']['failed']}\n")
            f.write(f"  Errors: {report['summary']['errors']}\n\n")
            
            if report['severity_breakdown']:
                f.write("Critical Issues by Severity:\n")
                for severity, count in report['severity_breakdown'].items():
                    f.write(f"  {severity}: {count}\n")
            
            f.write("\nFailed Tests:\n")
            for result in report['results']:
                if result['status'] == 'FAIL':
                    f.write(f"  [{result['severity']}] {result['test_name']}: {result['details']}\n")

async def main():
    """Main execution function."""
    try:
        # Initialize penetration tester
        pentester = HyperOptPenetrationTester()
        
        # Run all tests
        results = await pentester.run_all_tests()
        
        # Generate and save report
        pentester.save_report()
        
        # Print summary
        summary = pentester.generate_report()['summary']
        print(f"\nPenetration Testing Complete!")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed: {summary['passed']}")
        print(f"Failed: {summary['failed']}")
        print(f"Errors: {summary['errors']}")
        
        if summary['failed'] > 0:
            print(f"\n⚠️  {summary['failed']} security issues found!")
            return 1
        else:
            print("\n✅ No critical security issues found!")
            return 0
            
    except Exception as e:
        logger.error(f"Penetration testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code) 