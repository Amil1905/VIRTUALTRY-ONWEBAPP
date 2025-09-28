const fs = require('fs');
const path = require('path');

// Read images folder
const imagesFolder = './images/clothes'; // Adjust path as needed
const imageFiles = fs.readdirSync(imagesFolder)
  .filter(file => file.endsWith('.jpg') || file.endsWith('.png') || file.endsWith('.jpeg'))
  .sort();

// Generate JSON data
const clothesData = imageFiles.map((filename, index) => {
  const id = index + 1;
  const nameNumber = filename.split('_')[0]; // Gets "00064" from "00064_00.jpg"
  
  return {
    id: id,
    name: `Style Item ${nameNumber}`,
    brand: 'Fashion Brand',
    price: `£${(Math.random() * 50 + 15).toFixed(2)}`, // Random price between £15-65
    rating: +(Math.random() * 2 + 3).toFixed(1), // Random rating 3.0-5.0
    reviews: Math.floor(Math.random() * 2000 + 100), // Random reviews 100-2100
    image: `/images/clothes/${filename}`,
    images: [`/images/clothes/${filename}`],
    category: 'Clothing',
    colors: ['Black', 'White', 'Grey'],
    sizes: ['S', 'M', 'L', 'XL'],
    inStock: true,
    prime: Math.random() > 0.3, // 70% chance of prime
    description: `Stylish garment item ${nameNumber} for fashion-forward individuals.`,
    features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
  };
});

// Output as formatted JSON
console.log('export const clothesData = [');
clothesData.forEach((item, index) => {
  console.log('  {');
  Object.entries(item).forEach(([key, value]) => {
    const formattedValue = typeof value === 'string' ? `'${value}'` : 
                          Array.isArray(value) ? `[${value.map(v => `'${v}'`).join(', ')}]` :
                          value;
    console.log(`    ${key}: ${formattedValue},`);
  });
  console.log('  }' + (index < clothesData.length - 1 ? ',' : ''));
});
console.log('];');

// Also save to file
fs.writeFileSync('clothes-data.js', `export const clothesData = ${JSON.stringify(clothesData, null, 2)};`);
console.log('\nData also saved to clothes-data.js');