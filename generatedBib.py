import time
import requests
import json
import re
from urllib.parse import quote

references = [
    "Network Programmability and Automation Fundamentals",
    "Scalability, consistency, reliability and security in SDN controllers: a survey of diverse SDN controllers",
    "Artificial intelligence techniques for securing fog computing environments: trends, challenges, and future directions",
    "A Detailed Inspection of Machine Learning Based Intrusion Detection Systems for Software Defined Networks",
    "Distributed Software-Defined Networking Management: An Overview and Open Challenges",
    "The origin and evolution of open programmable networks and SDN",
    "Performance evaluation of different SDN controllers",
    "A flexible GraphQL northbound API for intent-based SDN applications",
    "Evaluating scalability, resiliency, and load balancing in software-defined networking",
    "Comprehensive survey of implementing multiple controllers in a Software-Defined Network (SDN)",
    "On the (in) security of the control plane of sdn architecture: A survey",
    "Survey on network virtualization hypervisors for software defined networking",
    "Network Security and Management in SDN",
    "A comprehensive survey of in-band control in sdn: Challenges and opportunities",
    "Performance analysis of python based SDN controllers over real internet topology",
    "A survey on the security of stateful SDN data planes",
    "A database approach to sdn control plane design",
    "Performance Analysis of Software Defined Network Concepts in Networked Embedded Systems",
    "A low latency, high throughput and scalable hardware architecture for flow tables in software defined networks",
    "A comprehensive survey on machine learning using in software defined networks (SDN)",
    "Software defined networking: a review on architecture, security and applications",
    "A survey on software defined networking and its applications",
    "Dynamic control plane for SDN at scale",
    "Comparison of software defined networking with traditional networking",
    "Comparative analysis of software defined network performance and conventional based on latency parameters",
    "A novel adaptive east–west interface for a heterogeneous and distributed sdn network",
    "A survey on large-scale software defined networking (SDN) testbeds: Approaches and challenges",
    "Software-defined networking: Categories, analysis, and future directions",
    "A survey on SDN and SDCN traffic measurement: Existing approaches and research challenges",
    "ML-based pre-deployment SDN performance prediction with neural network boosting regression",
    "An SDN-based network layer solution to improve the fairness and throughput of multi-path TCP",
    "SDN Controller for Optical Network Control",
    "AI and ML–Enablers for beyond 5G Networks",
    "AI/ML in Security Orchestration, Automation and Response: Future Research Directions",
    "Assessment of SDN Controllers in Wireless Environment Using a Multi-Criteria Technique",
    "A Comprehensive Review of Traffic Generators in Software-Defined Networking (SDN)",
    "Energy-saving traffic scheduling in backbone networks with software-defined networks",
    "How to realize the smooth transition from traditional network architecture to SDN",
    "Spider: A practical fuzzing framework to uncover stateful performance issues in sdn controllers",
    "Challenges of traditional networks and development of programmable networks",
    "ClassBench-ng: Benchmarking packet classification algorithms in the OpenFlow era",
    "Performance modelling and analysis of software-defined networking under bursty multimedia traffic",
    "Resource optimization in edge and SDN-based edge computing: a comprehensive study",
    "Effective flow table space management using policy-based routing approach in hybrid sdn network",
    "Constrained Reinforcement Learning for Adaptive Controller Synchronization in Distributed SDN",
    "Emulation and Analysis of Software-Defined Networks for the Detection of DDoS Attacks",
    "Performance analysis of software-defined networks (SDN) via POX controller simulation in Mininet",
    "SDN security review: Threat taxonomy, implications, and open challenges",
    "Analysis of software-defined networking (sdn) performance in wired and wireless networks across various topologies, including single, linear, and tree structures",
    "Flexible architecture for the future internet scalability of SDN control plane",
    "Latency and throughput advantage of leaf-enforced quality of service in software-defined networking for large traffic flows",
    "A comparative evaluation of ODL and ONOS controllers in software-defined network environments",
    "Impact of adaptive consistency on distributed sdn applications: An empirical study",
    "ITC: Intrusion tolerant controller for multicontroller SDN architecture",
    "Controller placement problem during SDN deployment in the ISP/Telco networks: A survey",
    "Security Enhancement through Flow-based Centralized control in SDN",
    "A qualitative and comparative performance assessment of logically centralized sdn controllers via mininet emulator",
    "Extensive performance analysis of OpenDayLight (ODL) and open network operating system (ONOS) SDN controllers",
    "Comparison of basic SDN controllers",
    "Software defined networking emulator for network application testing",
    "A survey on virtual network functions for media streaming: Solutions and future challenges",
    "A survey on trustworthy edge intelligence: From security and reliability to transparency and sustainability",
    "SDN improvements and solutions for traditional networks",
    "Comparative Analysis of Centralized and Distributed SDN Environments for IoT Networks",
    "SDN controllers: A comprehensive analysis and performance evaluation study"
]

class CrossRefBibTeXGenerator:
    def __init__(self):
        self.base_url = "https://api.crossref.org/works"
        self.headers = {
            'User-Agent': 'BibTeXGenerator/1.0 (mailto:researcher@example.com)',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def test_connection(self):
        """
        Test connection to CrossRef API before processing all references.
        """
        print("🔍 Testing connection to CrossRef API...")
        
        try:
            # Simple test query
            test_url = f"{self.base_url}?query=test&rows=1"
            response = self.session.get(test_url, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            if 'message' in data:
                print("✅ Connection successful!")
                return True
            else:
                print("❌ Unexpected response format")
                return False
                
        except requests.exceptions.Timeout:
            print("❌ Connection test timed out - you may have network connectivity issues")
            print("💡 Try checking your internet connection or using a VPN")
            return False
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to CrossRef API")
            print("💡 Check if you're behind a firewall or need proxy settings")
            return False
        except Exception as e:
            print(f"❌ Connection test failed: {str(e)}")
            return False
        """
        Search CrossRef API for a publication by title with retry logic.
        Returns a list of potential matches.
        """
        for attempt in range(max_retries):
            try:
                # Clean and encode the search query
                query = quote(title.strip())
                url = f"{self.base_url}?query.title={query}&rows={max_results}&sort=relevance"
                
                # Increase timeout and add retry logic
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                return data.get('message', {}).get('items', [])
                
            except requests.exceptions.Timeout:
                print(f"   ⏱️ Timeout on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"   ❌ Final timeout after {max_retries} attempts")
                    return []
            
            except requests.exceptions.ConnectionError as e:
                print(f"   🔌 Connection error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(10 * (attempt + 1))  # Longer wait for connection issues
                    continue
                else:
                    print(f"   ❌ Final connection error after {max_retries} attempts")
                    return []
            
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Rate limited
                    print(f"   🛑 Rate limited on attempt {attempt + 1}/{max_retries}")
                    if attempt < max_retries - 1:
                        time.sleep(60)  # Wait 1 minute for rate limit
                        continue
                    else:
                        print(f"   ❌ Rate limit exceeded after {max_retries} attempts")
                        return []
                else:
                    print(f"   ❌ HTTP error: {str(e)}")
                    return []
            
            except requests.exceptions.RequestException as e:
                print(f"   ❌ Request error on attempt {attempt + 1}/{max_retries}: {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                else:
                    return []
            
            except json.JSONDecodeError as e:
                print(f"   ❌ JSON decode error: {str(e)}")
                return []
        
        return []
    
    def calculate_similarity(self, original_title, found_title):
        """
        Calculate similarity between original and found titles.
        Simple word-based similarity metric.
        """
        original_words = set(original_title.lower().split())
        found_words = set(found_title.lower().split())
        
        if not original_words or not found_words:
            return 0.0
        
        intersection = original_words.intersection(found_words)
        union = original_words.union(found_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def find_best_match(self, original_title, search_results, min_similarity=0.3):
        """
        Find the best matching result based on title similarity.
        """
        best_match = None
        best_similarity = 0.0
        
        for item in search_results:
            found_titles = item.get('title', [])
            if not found_titles:
                continue
            
            found_title = found_titles[0]  # Use the first title
            similarity = self.calculate_similarity(original_title, found_title)
            
            if similarity > best_similarity and similarity >= min_similarity:
                best_similarity = similarity
                best_match = item
        
        return best_match, best_similarity
    
    def clean_string(self, text):
        """
        Clean strings for BibTeX format.
        """
        if not text:
            return ""
        
        # Remove or replace problematic characters
        text = str(text)
        text = text.replace('{', '').replace('}', '')
        text = text.replace('&', '\\&')
        text = text.replace('%', '\\%')
        text = text.replace('$', '\\$')
        text = text.replace('#', '\\#')
        text = text.replace('_', '\\_')
        text = text.replace('^', '\\^{}')
        
        return text.strip()
    
    def generate_bibtex_key(self, item, index):
        """
        Generate a unique BibTeX key for the entry.
        """
        # Try to use first author's last name
        authors = item.get('author', [])
        if authors and 'family' in authors[0]:
            author_key = re.sub(r'[^a-zA-Z0-9]', '', authors[0]['family'])[:10]
        else:
            author_key = "unknown"
        
        # Get year
        year = "0000"
        if 'published-print' in item:
            year = str(item['published-print']['date-parts'][0][0])
        elif 'published-online' in item:
            year = str(item['published-online']['date-parts'][0][0])
        elif 'created' in item:
            year = str(item['created']['date-parts'][0][0])
        
        return f"{author_key.lower()}{year}_{index:03d}"
    
    def crossref_to_bibtex(self, item, index, original_title):
        """
        Convert CrossRef JSON response to BibTeX format.
        """
        # Determine entry type
        entry_type = item.get('type', 'article')
        type_mapping = {
            'journal-article': 'article',
            'book-chapter': 'inbook',
            'book': 'book',
            'proceedings-article': 'inproceedings',
            'conference-paper': 'inproceedings',
            'thesis': 'phdthesis',
            'report': 'techreport'
        }
        bibtex_type = type_mapping.get(entry_type, 'article')
        
        # Generate key
        key = self.generate_bibtex_key(item, index)
        
        # Start BibTeX entry
        bibtex = f"@{bibtex_type}{{{key},\n"
        
        # Title
        titles = item.get('title', [])
        if titles:
            title = self.clean_string(titles[0])
            bibtex += f"  title={{{title}}},\n"
        
        # Authors
        authors = item.get('author', [])
        if authors:
            author_list = []
            for author in authors:
                given = author.get('given', '')
                family = author.get('family', '')
                if family:
                    if given:
                        author_list.append(f"{family}, {given}")
                    else:
                        author_list.append(family)
            
            if author_list:
                authors_str = " and ".join(author_list)
                bibtex += f"  author={{{self.clean_string(authors_str)}}},\n"
        
        # Journal/Container
        container_titles = item.get('container-title', [])
        if container_titles:
            container = self.clean_string(container_titles[0])
            if bibtex_type == 'article':
                bibtex += f"  journal={{{container}}},\n"
            elif bibtex_type in ['inproceedings', 'inbook']:
                bibtex += f"  booktitle={{{container}}},\n"
        
        # Volume
        volume = item.get('volume')
        if volume:
            bibtex += f"  volume={{{self.clean_string(volume)}}},\n"
        
        # Issue/Number
        issue = item.get('issue')
        if issue:
            bibtex += f"  number={{{self.clean_string(issue)}}},\n"
        
        # Pages
        page = item.get('page')
        if page:
            bibtex += f"  pages={{{self.clean_string(page)}}},\n"
        
        # Year
        year = None
        if 'published-print' in item:
            year = item['published-print']['date-parts'][0][0]
        elif 'published-online' in item:
            year = item['published-online']['date-parts'][0][0]
        elif 'created' in item:
            year = item['created']['date-parts'][0][0]
        
        if year:
            bibtex += f"  year={{{year}}},\n"
        
        # Publisher
        publisher = item.get('publisher')
        if publisher:
            bibtex += f"  publisher={{{self.clean_string(publisher)}}},\n"
        
        # DOI
        doi = item.get('DOI')
        if doi:
            bibtex += f"  doi={{{doi}}},\n"
        
        # URL
        url = item.get('URL')
        if url:
            bibtex += f"  url={{{url}}},\n"
        
        # Add note about original search title if different
        found_titles = item.get('title', [])
        if found_titles and found_titles[0].lower() != original_title.lower():
            bibtex += f"  note={{Original search: {self.clean_string(original_title)}}},\n"
        
        # Remove trailing comma and close entry
        bibtex = bibtex.rstrip(',\n') + '\n}\n'
        
        return bibtex
    
    def generate_fallback_entry(self, title, index):
        """
        Generate a basic BibTeX entry for references not found in CrossRef.
        """
        key = f"notfound_{index:03d}"
        
        bibtex = f"@misc{{{key},\n"
        bibtex += f"  title={{{self.clean_string(title)}}},\n"
        bibtex += f"  author={{Unknown}},\n"
        bibtex += f"  year={{Unknown}},\n"
        bibtex += f"  note={{Reference not found in CrossRef database - requires manual completion}}\n"
        bibtex += "}\n"
        
        return bibtex
    
    def generate_bibtex_from_references(self, output_file="crossref_references.bib"):
        """
        Generate BibTeX entries for all references using CrossRef API.
        """
        successful_entries = 0
        failed_entries = 0
        
        print(f"🚀 Starting CrossRef BibTeX generation...")
        print(f"📚 Total references to process: {len(references)}")
        print(f"📁 Output file: {output_file}")
        print("⏱️  Processing with polite delays...\n")
        
        with open(output_file, "w", encoding="utf-8") as bibfile:
            # Write header
            bibfile.write("% Auto-generated BibTeX file from SDN research references\n")
            bibfile.write("% Generated using CrossRef API\n")
            bibfile.write(f"% Total references processed: {len(references)}\n")
            bibfile.write(f"% Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for i, ref in enumerate(references, 1):
                print(f"[{i:2d}/{len(references)}] Processing: {ref[:60]}...")
                
                try:
                    # Search CrossRef
                    search_results = self.search_crossref(ref)
                    
                    if search_results:
                        # Find best match
                        best_match, similarity = self.find_best_match(ref, search_results)
                        
                        if best_match:
                            # Generate BibTeX entry
                            bibtex_entry = self.crossref_to_bibtex(best_match, i, ref)
                            
                            # Write to file with metadata
                            bibfile.write(f"% Reference {i}: {ref}\n")
                            bibfile.write(f"% Similarity: {similarity:.2f}\n")
                            if 'title' in best_match and best_match['title']:
                                bibfile.write(f"% Found: {best_match['title'][0]}\n")
                            bibfile.write(bibtex_entry + "\n")
                            
                            successful_entries += 1
                            print(f"   ✅ Found match (similarity: {similarity:.2f})")
                        else:
                            # No good match found
                            bibfile.write(f"% Reference {i}: {ref}\n")
                            bibfile.write(f"% No suitable match found (similarity too low)\n")
                            fallback_entry = self.generate_fallback_entry(ref, i)
                            bibfile.write(fallback_entry + "\n")
                            
                            failed_entries += 1
                            print(f"   ❌ No suitable match found")
                    else:
                        # No search results
                        bibfile.write(f"% Reference {i}: {ref}\n")
                        bibfile.write(f"% No results found in CrossRef\n")
                        fallback_entry = self.generate_fallback_entry(ref, i)
                        bibfile.write(fallback_entry + "\n")
                        
                        failed_entries += 1
                        print(f"   ❌ No results found")
                
                except Exception as e:
                    # Handle unexpected errors
                    bibfile.write(f"% Reference {i}: {ref}\n")
                    bibfile.write(f"% ERROR: {str(e)}\n")
                    fallback_entry = self.generate_fallback_entry(ref, i)
                    bibfile.write(fallback_entry + "\n")
                    
                    failed_entries += 1
                    print(f"   ❌ Error: {str(e)}")
                
                # Polite delay to respect CrossRef's rate limits and avoid timeouts
                time.sleep(2)  # Increased delay to reduce timeout issues
        
        # Print summary
        print(f"\n📊 Processing Complete!")
        print(f"✅ Successfully processed: {successful_entries}")
        print(f"❌ Failed to process: {failed_entries}")
        print(f"📈 Success rate: {(successful_entries / len(references)) * 100:.1f}%")
        print(f"📁 BibTeX file saved as: {output_file}")
        
        return successful_entries, failed_entries
    
    def validate_and_report(self, bibfile_path="crossref_references.bib"):
        """
        Validate the generated BibTeX file and provide a quality report.
        """
        print(f"\n🔍 Validating BibTeX file: {bibfile_path}")
        
        try:
            with open(bibfile_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Count entries
            total_entries = content.count('@')
            misc_entries = content.count('@misc')
            article_entries = content.count('@article')
            inproceedings_entries = content.count('@inproceedings')
            
            # Count entries with DOIs
            doi_entries = content.count('doi=')
            
            # Count entries needing manual work
            manual_entries = content.count('requires manual completion')
            
            print(f"📊 Validation Report:")
            print(f"   Total entries: {total_entries}")
            print(f"   Articles: {article_entries}")
            print(f"   Conference papers: {inproceedings_entries}")
            print(f"   Misc/Other: {misc_entries}")
            print(f"   Entries with DOI: {doi_entries}")
            print(f"   Entries needing manual work: {manual_entries}")
            print(f"   Quality score: {((total_entries - manual_entries) / total_entries * 100):.1f}%")
            
        except FileNotFoundError:
            print(f"❌ File not found: {bibfile_path}")
        except Exception as e:
            print(f"❌ Validation error: {str(e)}")

def main():
    """
    Main function to run the CrossRef BibTeX generator.
    """
    generator = CrossRefBibTeXGenerator()
    
    try:
        # Generate BibTeX entries
        successful, failed = generator.generate_bibtex_from_references()
        
        # Validate results
        generator.validate_and_report()
        
        # Provide recommendations
        print(f"\n💡 Recommendations:")
        if failed > 0:
            print(f"   - Review {failed} failed entries and complete manually")
            print(f"   - Search for missing papers using DOI or exact titles")
        
        print(f"   - Verify author names and publication details")
        print(f"   - Check for duplicate entries")
        print(f"   - Validate DOI links")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Process interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {str(e)}")

if __name__ == "__main__":
    main()