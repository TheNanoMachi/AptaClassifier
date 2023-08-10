# Import scrapy
import scrapy
# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess

# Create the Spider class
class Aptagen_Spider(scrapy.Spider):
  name = "aptagen_spider"
  crawled = 0
  # start_requests method
  def start_requests(self):
    # urls = 'https://www.aptagen.com/apta-index/?pageNumber=1&targetCategory=All&sortBy=Length+low-high&aptazyme=None'
    # urls = 'https://www.aptagen.com/apta-index/?pageNumber=1&targetCategory=All&affinityMin=&affinityMax=&affinityUnits=&aptamerChemistry=DNA&sortBy=Length+(low-high)&aptazyme=None'
    urls = 'https://www.aptagen.com/apta-index/?pageNumber=1&targetCategory=Small+Organic&affinityMin=&affinityMax=&affinityUnits=&aptamerChemistry=All&sortBy=Length+(low-high)&aptazyme=None&searchQuery='
    # urls = 'https://www.aptagen.com/apta-index/?pageNumber=1&targetCategory=All&affinityMin=&affinityMax=&affinityUnits=&aptamerChemistry=All&sortBy=Length+(low-high)&aptazyme=None&pageSize=10&searchQuery='
    yield scrapy.Request(url = urls, callback = self.parse)

 # Parsing the front page
  def parse(self, response):
    Apta_links = response.xpath('//div[@class = "row result"]//a/@href')
    links_to_follow = Apta_links.extract()
    for url in links_to_follow:
      yield response.follow(url = url, callback = self.parse_pages)

  def parse_pages(self, response):
    Sequence = response.css('span.apta-sequence>*::text').extract()
    for i in range(len(Sequence)):
      if len(Sequence[i]) == 3:
        x = ""
        for j in range(3):
          if Sequence[i][j] in ["A", "G", "T", "C", "U"]:
            x = Sequence[i][j]
        Sequence[i] = x
    Reference = response.xpath("/html/body/div[1]/div/div[2]/div[3]/div[2]/div/div[3]/div[2]/h3/text()").extract()
    # Name = response.xpath('//div[@class = "aptamer-details"]/h3//span[@itemprop = "name"]/text()').extract()
    Target = response.xpath('/html/body/div[1]/div/div[2]/div[3]/div[2]/div/span[3]/text()').extract()
    Info = response.xpath('//div[@class = "aptamer-details"]/label/text() | //div[@class = "aptamer-details"]/span/text() | //div[@class = "aptamer-details"]/span[6]/sub/text()').extract()
    SequenceInfo = response.xpath('//div[@class = "bottom"]/label[position() > 1 and position() < last()]/text() | //div[@class = "bottom"]/span[position() > 1]/text()').extract()
    sequence_text = ""
    for i in tuple(Sequence):
      sequence_text += i
    Apta_dict[sequence_text] = ["{{"] + Reference + ["}}"] + ["{{"] + Target + ["}}"] + ["{{"] + Info + ["}}"] + ["{{"] + SequenceInfo + ["}}"]
    

# Initialize the dictionary **outside** of the Spider class
Apta_dict = dict()

# Run the Spider
process = CrawlerProcess()
process.crawl(Aptagen_Spider)
process.start()

with open('aptagen_sequences_dna_only-1.txt', 'w+', encoding='utf-8') as result:
  for k, v in Apta_dict.items():
    result.write("{{ ")
    result.write(k[2:(len(k)-2)])
    result.write(" }}")
    result.write(" ")
    result.write(" ".join(v))
    result.write(' || ')