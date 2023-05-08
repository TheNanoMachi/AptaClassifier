# Import scrapy
import scrapy
# Import the CrawlerProcess: for running the spider
from scrapy.crawler import CrawlerProcess

# Create the Spider class
class Aptagen_Spider(scrapy.Spider):
  name = "aptagen_spider"
  # start_requests method
  def start_requests(self):
    urls = 'https://www.aptagen.com/apta-index/?pageNumber=1&targetCategory=All&sortBy=Length+low-high&aptazyme=None'
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
        Sequence[i] = Sequence[i][1]
    Name = response.xpath('//div[@class = "aptamer-details"]/h3//span[@itemprop = "name"]/text()').extract()
    # AptaChem = response.xpath('//div[@class = "aptamer-details"]/span[2]/text()').extract()
    # Target = response.xpath('//div[@class = "aptamer-details"]/span[3]/text()').extract()
    # TargetCategory = response.xpath('//div[@class = "aptamer-details"]/span[4]/text()').extract()
    # Kd = response.xpath('//div[@class = "aptamer-details"]/span[5]/text()').extract()
    # Buffer = response.xpath('//div[@class = "aptamer-details"]/span[6]/text() | //div[@class = "aptamer-details"]/span[6]/sub/text()').extract()
    # Temp = response.xpath('//div[@class = "aptamer-details"]/span[7]/text()').extract()
    Info = response.xpath('//div[@class = "aptamer-details"]/label/text() | //div[@class = "aptamer-details"]/span/text() | //div[@class = "aptamer-details"]/span[6]/sub/text()').extract()
    # Length = response.xpath('//div[@class = "bottom"]/span[2]/text()').extract()
    # MW = response.xpath('//div[@class = "bottom"]/span[3]/text()').extract()
    # ExtCoeff = response.xpath('//div[@class = "bottom"]/span[4]/text()').extract()
    # GC = response.xpath('//div[@class = "bottom"]/span[5]/text()').extract()
    # nmolesOD260 = response.xpath('//div[@class = "bottom"]/span[6]/text()').extract()
    # ugOD260 = response.xpath('//div[@class = "bottom"]/span[7]/text()').extract()
    # Apta_dict[tuple(Sequence)] = Name + AptaChem + Target + TargetCategory + Kd + Buffer + Temp + ['Extra Details-->'] + Length + MW + ExtCoeff + GC + nmolesOD260 + ugOD260
    SequenceInfo = response.xpath('//div[@class = "bottom"]/label[position() > 1 and position() < last()]/text() | //div[@class = "bottom"]/span[position() > 1]/text()').extract()
    Apta_dict[tuple(Sequence)] = Name + Info + SequenceInfo
    

# Initialize the dictionary **outside** of the Spider class
Apta_dict = dict()

# Run the Spider
process = CrawlerProcess()
process.crawl(Aptagen_Spider)
process.start()

# for keys,values in Apta_dict.items():
#     print(keys, values) 

with open('result.txt', 'w+', encoding='utf-8') as result:
  for k, v in Apta_dict.items():
    result.write(", ".join(k))
    result.write(", ".join(v))
    result.write('\n\n')
  result.close()