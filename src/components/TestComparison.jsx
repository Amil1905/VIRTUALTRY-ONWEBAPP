import React, { useState, useRef, useCallback, useEffect } from 'react';
import { 
  Upload, User, ShoppingBag, Star, Filter, Search, 
  ChevronDown, Heart, Download, BarChart3, Loader, 
  Camera, RefreshCw, Zap, Brain, AlertCircle, CheckCircle,
  TrendingUp, TrendingDown, Minus, Settings, Info, 
  Layout, Grid, List, Eye, Share2, BookOpen, Award, Sparkles,
  ArrowLeft, Plus, ShoppingCart, Truck, Shield, RotateCcw,
  MapPin, Clock, Palette, Shirt, MessageSquare
} from 'lucide-react';

const TryOnApp = () => {
  const [currentView, setCurrentView] = useState('catalog');
  const [selectedGarment, setSelectedGarment] = useState(null);
  const [userImage, setUserImage] = useState(null);
  const [userImageFile, setUserImageFile] = useState(null);
  const [tryOnResults, setTryOnResults] = useState({});
  const [loading, setLoading] = useState({});
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [selectedImage, setSelectedImage] = useState(0);
  const [selectedSize, setSelectedSize] = useState('M');
  const [selectedColor, setSelectedColor] = useState(0);
  const [quantity, setQuantity] = useState(1);
  
  // PromptDresser specific states
  const [stylePrompts, setStylePrompts] = useState(null);
  const [selectedStyleVariation, setSelectedStyleVariation] = useState(0);
  
  const fileInputRef = useRef(null);
  
  // API endpoints - PromptDresser style endpoint eklendi
  const API_ENDPOINTS = {
    'viton-hd': 'https://621ed3ca5a7f.ngrok-free.app/tryon', 
    'idm-vton': 'https://15e7ba87fe42.ngrok-free.app/try-on',
    'promptdresser': 'https://95625ace66dd.ngrok-free.app/tryon-style'  // Buraya yeni ngrok URL'inizi yazın
  };
  
  const models = [
    { 
      id: 'viton-hd', 
      name: 'VITON-HD', 
      type: 'GAN-based',
      description: 'High-definition virtual try-on',
      icon: <Zap className="w-4 h-4" />,
      color: 'bg-blue-500'
    },
    { 
      id: 'idm-vton', 
      name: 'IDM-VTON', 
      type: 'Diffusion-based',
      description: 'Ultra-realistic textures',
      icon: <Sparkles className="w-4 h-4" />,
      color: 'bg-purple-500'
    },
    { 
      id: 'promptdresser', 
      name: 'PromptDresser', 
      type: 'Text-guided',
      description: 'Style-aware with custom prompts',
      icon: <Shirt className="w-4 h-4" />,
      color: 'bg-red-500'
    }
  ];

  const categories = [
    'All', 'Shirts', 'T-Shirts', 'Polo', 'Hoodies', 'Jackets'
  ];

  const garments = [
    { 
      id: 1, 
      name: 'Premium Cotton T-Shirt', 
      brand: 'Amazon Essentials',
      price: '£24.99', 
      originalPrice: '£34.99',
      rating: 4.3,
      reviews: 3847,
      image: '/images/clothes/01430_00.jpg',
      images: ['/images/clothes/01430_00.jpg', '/images/clothes/01430_01.jpg'],
      category: 'T-Shirts',
      colors: ['Black', 'White', 'Navy', 'Grey'],
      sizes: ['XS', 'S', 'M', 'L', 'XL', 'XXL'],
      inStock: true,
      prime: true,
      description: 'Comfortable 100% cotton t-shirt perfect for everyday wear.',
      features: ['100% Cotton', 'Machine Washable', 'Preshrunk', 'Tagless']
    },
    { 
      id: 2, 
      name: 'Classic Polo Shirt', 
      brand: 'Goodthreads',
      price: '£18.50', 
      rating: 4.1,
      reviews: 2156,
      image: '/images/clothes/00064_00.jpg',
      images: ['/images/clothes/00064_00.jpg'],
      category: 'Polo',
      colors: ['White', 'Black', 'Navy', 'Red'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Classic fit polo shirt with modern styling.',
      features: ['Cotton Blend', 'Classic Fit', 'Ribbed Collar', 'Machine Wash']
    },
    { 
      id: 3, 
      name: 'Casual Button Shirt', 
      brand: 'Daily Ritual',
      price: '£32.00', 
      originalPrice: '£45.00',
      rating: 4.4,
      reviews: 1823,
      image: '/images/clothes/00145_00.jpg',
      images: ['/images/clothes/00145_00.jpg'],
      category: 'Shirts',
      colors: ['Blue', 'White', 'Black'],
      sizes: ['S', 'M', 'L', 'XL', 'XXL'],
      inStock: true,
      prime: true,
      description: 'Versatile button-up shirt for work or casual wear.',
      features: ['Wrinkle Resistant', 'Regular Fit', 'Cotton Blend', 'Easy Care']
    },
    { 
      id: 4, 
      name: 'Lightweight Hoodie', 
      brand: 'Core 10',
      price: '£29.99', 
      rating: 4.2,
      reviews: 5432,
      image: '/images/clothes/06893_00.jpg',
      images: ['/images/clothes/06893_00.jpg'],
      category: 'Hoodies',
      colors: ['Grey', 'Black', 'Navy'],
      sizes: ['XS', 'S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Comfortable lightweight hoodie for layering.',
      features: ['Cotton-Poly Blend', 'Kangaroo Pocket', 'Adjustable Hood', 'Ribbed Cuffs']
    },
    // Ek kıyafetler - JSON'unuzda olan diğer kıyafetleri buraya ekleyebilirsiniz
    { 
      id: 5, 
      name: 'Style Test Shirt 1', 
      brand: 'Test Brand',
      price: '£25.99', 
      rating: 4.0,
      reviews: 1500,
      image: '/images/clothes/07309_00.jpg',  // JSON'unuzda olan dosya adı
      images: ['/images/clothes/07309_00.jpg'],
      category: 'Shirts',
      colors: ['Blue', 'White'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Test garment for style variations.',
      features: ['Cotton', 'Style Testing', 'Multiple Fits']
    },
    { 
      id: 6, 
      name: 'Style Test Shirt 2', 
      brand: 'Test Brand',
      price: '£27.99', 
      rating: 4.1,
      reviews: 1200,
      image: '/images/clothes/01215_00.jpg',  // JSON'unuzda olan dosya adı
      images: ['/images/clothes/01215_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'Grey'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Another test garment for style variations.',
      features: ['Cotton', 'Style Testing', 'Tucking Options']
    },
    { 
      id: 7, 
      name: 'Style Test Shirt 2', 
      brand: 'Test Brand',
      price: '£27.99', 
      rating: 4.1,
      reviews: 1200,
      image: '/images/clothes/06894_00.jpg',  // JSON'unuzda olan dosya adı
      images: ['/images/clothes/06894_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'Grey'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Another test garment for style variations.',
      features: ['Cotton', 'Style Testing', 'Tucking Options']
    },
    {
      id: 9,
      name: 'Style Item 09505',
      brand: 'Core 10',
      price: '£51.04',
      rating: 5.0,
      reviews: 798,
      image: '/images/clothes/09505_00.jpg',
      images: ['/images/clothes/09505_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 09505 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 10,
      name: 'Style Item 09523',
      brand: 'Peak Velocity',
      price: '£45.58',
      rating: 4.0,
      reviews: 626,
      image: '/images/clothes/09523_00.jpg',
      images: ['/images/clothes/09523_00.jpg'],
      category: 'T-Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 09523 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 11,
      name: 'Style Item 09588',
      brand: 'Peak Velocity',
      price: '£59.87',
      rating: 4.8,
      reviews: 1592,
      image: '/images/clothes/09588_00.jpg',
      images: ['/images/clothes/09588_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 09588 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 12,
      name: 'Style Item 09674',
      brand: 'Amazon Essentials',
      price: '£57.74',
      rating: 4.4,
      reviews: 1667,
      image: '/images/clothes/09674_00.jpg',
      images: ['/images/clothes/09674_00.jpg'],
      category: 'Polo',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 09674 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 13,
      name: 'Style Item 09680',
      brand: 'Buttoned Down',
      price: '£54.67',
      rating: 4.0,
      reviews: 236,
      image: '/images/clothes/09680_00.jpg',
      images: ['/images/clothes/09680_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 09680 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 14,
      name: 'Style Item 09869',
      brand: 'Core 10',
      price: '£49.13',
      rating: 4.3,
      reviews: 431,
      image: '/images/clothes/09869_00.jpg',
      images: ['/images/clothes/09869_00.jpg'],
      category: 'Hoodies',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 09869 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 15,
      name: 'Style Item 09987',
      brand: 'Daily Ritual',
      price: '£35.36',
      rating: 3.6,
      reviews: 1583,
      image: '/images/clothes/09987_00.jpg',
      images: ['/images/clothes/09987_00.jpg'],
      category: 'Jackets',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 09987 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 16,
      name: 'Style Item 10070',
      brand: 'Peak Velocity',
      price: '£43.39',
      rating: 4.1,
      reviews: 405,
      image: '/images/clothes/10070_00.jpg',
      images: ['/images/clothes/10070_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10070 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 17,
      name: 'Style Item 10094',
      brand: 'Core 10',
      price: '£50.86',
      rating: 4.2,
      reviews: 1250,
      image: '/images/clothes/10094_00.jpg',
      images: ['/images/clothes/10094_00.jpg'],
      category: 'Jackets',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10094 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 18,
      name: 'Style Item 10116',
      brand: 'Peak Velocity',
      price: '£57.19',
      rating: 4.6,
      reviews: 565,
      image: '/images/clothes/10116_00.jpg',
      images: ['/images/clothes/10116_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10116 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 19,
      name: 'Style Item 10380',
      brand: 'Goodthreads',
      price: '£22.05',
      rating: 4.4,
      reviews: 761,
      image: '/images/clothes/10380_00.jpg',
      images: ['/images/clothes/10380_00.jpg'],
      category: 'Jackets',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10380 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 20,
      name: 'Style Item 10432',
      brand: 'Peak Velocity',
      price: '£42.71',
      rating: 4.3,
      reviews: 987,
      image: '/images/clothes/10432_00.jpg',
      images: ['/images/clothes/10432_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10432 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 21,
      name: 'Style Item 10493',
      brand: 'Daily Ritual',
      price: '£25.23',
      rating: 5.0,
      reviews: 409,
      image: '/images/clothes/10493_00.jpg',
      images: ['/images/clothes/10493_00.jpg'],
      category: 'Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10493 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 22,
      name: 'Style Item 10494',
      brand: 'Amazon Essentials',
      price: '£57.54',
      rating: 3.6,
      reviews: 1588,
      image: '/images/clothes/10494_00.jpg',
      images: ['/images/clothes/10494_00.jpg'],
      category: 'Polo',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10494 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 23,
      name: 'Style Item 10543',
      brand: 'Peak Velocity',
      price: '£23.11',
      rating: 4.6,
      reviews: 1437,
      image: '/images/clothes/10543_00.jpg',
      images: ['/images/clothes/10543_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10543 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 24,
      name: 'Style Item 10545',
      brand: 'Peak Velocity',
      price: '£37.26',
      rating: 4.4,
      reviews: 760,
      image: '/images/clothes/10545_00.jpg',
      images: ['/images/clothes/10545_00.jpg'],
      category: 'Polo',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10545 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 25,
      name: 'Style Item 10579',
      brand: 'Daily Ritual',
      price: '£34.09',
      rating: 3.7,
      reviews: 743,
      image: '/images/clothes/10579_00.jpg',
      images: ['/images/clothes/10579_00.jpg'],
      category: 'T-Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10579 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 26,
      name: 'Style Item 10640',
      brand: 'Buttoned Down',
      price: '£36.08',
      rating: 5.0,
      reviews: 338,
      image: '/images/clothes/10640_00.jpg',
      images: ['/images/clothes/10640_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10640 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 27,
      name: 'Style Item 10731',
      brand: 'Peak Velocity',
      price: '£27.54',
      rating: 4.8,
      reviews: 1195,
      image: '/images/clothes/10731_00.jpg',
      images: ['/images/clothes/10731_00.jpg'],
      category: 'Hoodies',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10731 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 28,
      name: 'Style Item 10760',
      brand: 'Core 10',
      price: '£60.0',
      rating: 3.9,
      reviews: 1580,
      image: '/images/clothes/10760_00.jpg',
      images: ['/images/clothes/10760_00.jpg'],
      category: 'T-Shirts',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: true,
      description: 'Stylish garment item 10760 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    },
    {
      id: 29,
      name: 'Style Item 10838',
      brand: 'Core 10',
      price: '£36.67',
      rating: 4.0,
      reviews: 1552,
      image: '/images/clothes/10838_00.jpg',
      images: ['/images/clothes/10838_00.jpg'],
      category: 'Sweaters',
      colors: ['Black', 'White', 'Grey', 'Navy'],
      sizes: ['S', 'M', 'L', 'XL'],
      inStock: true,
      prime: false,
      description: 'Stylish garment item 10838 for modern fashion.',
      features: ['Premium Quality', 'Comfortable Fit', 'Durable Material']
    }
    
  ];

  // PromptDresser style prompts yükle
  const loadStylePrompts = useCallback(async (garment) => {
    try {
      const filename = garment.image.split('/').pop();
      // BURAYA YENİ NGROK URL'İNİZİ YAZIN
      const response = await fetch(`https://95625ace66dd.ngrok-free.app/get-prompts/${filename}`);
      
      if (response.ok) {
        const data = await response.json();
        if (!data.error) {
          setStylePrompts(data);
          console.log('Style prompts loaded:', data);
        } else {
          console.log('Style prompts error:', data.error);
          // Fallback - default style variations
          setStylePrompts({
            style_variations: [
              {
                name: "Casual & Relaxed",
                description: "Comfortable everyday look",
                tucking: "untucked",
                fit: "relaxed",
                sleeve_rolling: "short sleeve"
              },
              {
                name: "Smart Casual", 
                description: "Polished but comfortable",
                tucking: "french tucked",
                fit: "regular fit",
                sleeve_rolling: "a long-sleeved with the sleeves down"
              },
              {
                name: "Formal & Fitted",
                description: "Sharp and professional", 
                tucking: "fully tucked in",
                fit: "tight fit",
                sleeve_rolling: "a long-sleeved with the sleeves down"
              }
            ]
          });
        }
      }
    } catch (error) {
      console.log('Could not load style prompts:', error);
      // Fallback - default style variations
      setStylePrompts({
        style_variations: [
          {
            name: "Casual & Relaxed",
            description: "Comfortable everyday look",
            tucking: "untucked",
            fit: "relaxed", 
            sleeve_rolling: "short sleeve"
          },
          {
            name: "Smart Casual",
            description: "Polished but comfortable",
            tucking: "french tucked", 
            fit: "regular fit",
            sleeve_rolling: "a long-sleeved with the sleeves down"
          },
          {
            name: "Formal & Fitted",
            description: "Sharp and professional",
            tucking: "fully tucked in",
            fit: "tight fit",
            sleeve_rolling: "a long-sleeved with the sleeves down"
          }
        ]
      });
    }
  }, []);

  const handleUserImageUpload = useCallback((event) => {
    const file = event.target.files[0];
    if (file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file');
        return;
      }
      if (file.size > 10 * 1024 * 1024) {
        alert('Image size should be less than 10MB');
        return;
      }
      
      setUserImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setUserImage(e.target.result);
      reader.readAsDataURL(file);
    }
  }, []);

  const handleGarmentClick = (garment) => {
    setSelectedGarment(garment);
    setCurrentView('product');
    setSelectedImage(0);
    setSelectedColor(0);
    setSelectedStyleVariation(0); // Reset style selection
    loadStylePrompts(garment); // Style prompts yükle
  };

  const parseCompleteMetrics = (metricsData) => {
    try {
      if (typeof metricsData === 'string') {
        const parsed = JSON.parse(metricsData);
        return parsed;
      }
      return metricsData;
    } catch (error) {
      console.error('Error parsing metrics:', error);
      return null;
    }
  };

  const tryOnWithModel = async (garment, modelId) => {
    if (!userImageFile || !garment) {
      alert("Please upload your photo and select an item!");
      return;
    }

    setLoading(prev => ({ ...prev, [modelId]: true }));

    try {
      const formData = new FormData();
      formData.append('person_image', userImageFile);

      const response = await fetch(garment.image);
      if (!response.ok) throw new Error('Failed to fetch clothing image');
      
      const blob = await response.blob();
      const fileName = garment.image.split('/').pop();
      const file = new File([blob], fileName, { type: blob.type });
      formData.append('cloth_image', file);

      // PromptDresser için özel parametreler
      if (modelId === 'promptdresser') {
        if (stylePrompts && stylePrompts.style_variations) {
          const selectedVariation = stylePrompts.style_variations[selectedStyleVariation];
          
          const styleOverrides = {
            "tucking style": selectedVariation.tucking,
            "fit of upper cloth": selectedVariation.fit,
            "sleeve rolling style": selectedVariation.sleeve_rolling
          };
          
          formData.append('style_overrides', JSON.stringify(styleOverrides));
          
          console.log('PromptDresser style params:', styleOverrides);
        }
      }

      const apiResponse = await fetch(API_ENDPOINTS[modelId], {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        throw new Error(`API Error: ${apiResponse.status}`);
      }

      let metrics = {};
      let completeMetrics = null;

      if (modelId === 'viton-hd' || modelId === 'idm-vton') {
        const metricsHeader = apiResponse.headers.get('X-Metrics');
        if (metricsHeader) {
          completeMetrics = parseCompleteMetrics(metricsHeader);
          if (completeMetrics) {
            metrics = {
              ssim: completeMetrics.visual_quality?.ssim || 'N/A',
              lpips: completeMetrics.visual_quality?.lpips || 'N/A',
              psnr: completeMetrics.visual_quality?.psnr || 'N/A',
              fid: completeMetrics.visual_quality?.fid || 'N/A',
              is: completeMetrics.visual_quality?.is || 'N/A',
              quality_grade: completeMetrics.visual_quality?.quality_grade || 'N/A',
              inference_time: completeMetrics.performance?.inference_time_sec || 'N/A',
              memory_usage: completeMetrics.performance?.memory_usage_mb || 'N/A',
              gpu_memory: completeMetrics.performance?.gpu_memory_mb || 'N/A',
              model_size: completeMetrics.performance?.model_size_mb || 'N/A',
              cpu_usage: completeMetrics.performance?.cpu_usage_percent || 'N/A',
              overall_score: completeMetrics.summary?.overall_score || 'N/A',
              efficiency_rating: completeMetrics.summary?.efficiency_rating || 'N/A',
              resolution: completeMetrics.output_metrics?.output_resolution || 'N/A',
              file_size: completeMetrics.output_metrics?.file_size_mb || 'N/A'
            };
          }
        }
      }
      
      if (!completeMetrics) {
        metrics = {
          ssim: (0.7 + Math.random() * 0.3).toFixed(3),
          lpips: (Math.random() * 0.3).toFixed(3),
          psnr: (25 + Math.random() * 10).toFixed(1),
          inference_time: (1.5 + Math.random() * 3).toFixed(1),
          fid: (20 + Math.random() * 30).toFixed(1)
        };
      }

      const resultBlob = await apiResponse.blob();
      const resultUrl = URL.createObjectURL(resultBlob);

      setTryOnResults(prev => ({
        ...prev,
        [modelId]: {
          image: resultUrl,
          metrics: metrics,
          completeMetrics: completeMetrics,
          timestamp: new Date().toISOString(),
          garment: garment,
          // PromptDresser için kullanılan parametreleri kaydet
          ...(modelId === 'promptdresser' && stylePrompts && {
            usedStyleVariation: selectedStyleVariation,
            usedStyle: stylePrompts.style_variations[selectedStyleVariation]?.name || 'Default'
          })
        }
      }));

    } catch (error) {
      console.error(`Try-on failed for ${modelId}:`, error);
      alert(`${modelId} model error: ${error.message}`);
    } finally {
      setLoading(prev => ({ ...prev, [modelId]: false }));
    }
  };

  const filteredGarments = garments.filter(garment => {
    const matchesSearch = garment.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         garment.brand.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory = selectedCategory === 'all' || 
      garment.category.toLowerCase() === selectedCategory.toLowerCase();
    return matchesSearch && matchesCategory;
  });

  // Catalog View (Amazon-style grid)
  if (currentView === 'catalog') {
    return (
      <div className="min-h-screen bg-white">
        {/* Header */}
        <header className="bg-gray-900 text-white">
          <div className="px-4 py-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <div className="text-xl font-bold">amazon</div>
                <div className="text-sm">.co.uk</div>
              </div>
              <div className="flex-1 max-w-2xl mx-4">
                <div className="flex">
                  <select className="bg-gray-200 text-gray-800 px-3 py-2 rounded-l border-r border-gray-300">
                    <option>All Categories</option>
                  </select>
                  <input
                    type="text"
                    placeholder="Search Amazon.co.uk"
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="flex-1 px-4 py-2 text-gray-800"
                  />
                  <button className="bg-orange-400 hover:bg-orange-500 px-4 py-2 rounded-r">
                    <Search size={20} />
                  </button>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <div className="text-sm">
                  <div>Hello, Sign in</div>
                  <div className="font-bold">Account & Lists</div>
                </div>
                <div className="text-sm">
                  <div>Returns</div>
                  <div className="font-bold">& Orders</div>
                </div>
                <div className="flex items-center">
                  <ShoppingCart size={24} />
                  <span className="font-bold">Basket</span>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-gray-800 px-4 py-2">
            <div className="flex items-center space-x-6 text-sm">
              <span className="font-bold">All</span>
              <span>Today's Deals</span>
              <span>Customer Service</span>
              <span>Prime</span>
              <span>Fashion</span>
              <span>Electronics</span>
              <span>Home & Garden</span>
            </div>
          </div>
        </header>

        {/* Main Content - Catalog */}
        <div className="max-w-7xl mx-auto px-4 py-6">
          {/* Category filters */}
          <div className="mb-6">
            <div className="flex items-center space-x-4 mb-4">
              <h2 className="text-lg font-semibold">Clothing</h2>
              <div className="flex space-x-2">
                {categories.map(category => (
                  <button
                    key={category}
                    onClick={() => setSelectedCategory(category.toLowerCase())}
                    className={`px-4 py-2 text-sm border rounded ${
                      selectedCategory === category.toLowerCase()
                        ? 'bg-orange-100 border-orange-400 text-orange-800'
                        : 'bg-white border-gray-300 hover:bg-gray-50'
                    }`}
                  >
                    {category}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Upload Photo Section */}
          {!userImage ? (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6 text-center">
              <Camera size={48} className="text-blue-500 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-blue-900 mb-2">Try Virtual Fashion</h3>
              <p className="text-blue-700 mb-4">Upload your photo to see how clothes look on you with AI technology</p>
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleUserImageUpload}
                accept="image/*"
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
              >
                Upload Your Photo
              </button>
            </div>
          ) : (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
              <div className="flex items-center space-x-4">
                <img src={userImage} alt="Your photo" className="w-12 h-12 rounded-full object-cover" />
                <div>
                  <h3 className="font-medium text-green-900">Photo uploaded successfully!</h3>
                  <p className="text-green-700 text-sm">Click on any item to see virtual try-on with PromptDresser styles</p>
                </div>
                <button
                  onClick={() => {
                    setUserImage(null);
                    setUserImageFile(null);
                    setTryOnResults({});
                  }}
                  className="ml-auto text-green-600 hover:text-green-800"
                >
                  Change Photo
                </button>
              </div>
            </div>
          )}

          {/* Products Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
            {filteredGarments.map((garment) => (
              <div
                key={garment.id}
                onClick={() => handleGarmentClick(garment)}
                className="bg-white border border-gray-200 rounded-lg overflow-hidden hover:shadow-lg transition-shadow cursor-pointer"
              >
                <div className="aspect-square relative">
                  <img
                    src={garment.image}
                    alt={garment.name}
                    className="w-full h-full object-cover"
                  />
                  {garment.prime && (
                    <div className="absolute top-2 right-2 bg-blue-600 text-white text-xs px-2 py-1 rounded">
                      Prime
                    </div>
                  )}
                  {userImage && (
                    <div className="absolute top-2 left-2 bg-red-600 text-white text-xs px-2 py-1 rounded flex items-center">
                      <Shirt size={12} className="mr-1" />
                      Style Test
                    </div>
                  )}
                </div>
                
                <div className="p-4">
                  <h3 className="font-medium text-gray-900 text-sm mb-1 line-clamp-2">{garment.name}</h3>
                  <p className="text-gray-600 text-xs mb-2">{garment.brand}</p>
                  
                  <div className="flex items-center mb-2">
                    <div className="flex items-center">
                      {[...Array(5)].map((_, i) => (
                        <Star 
                          key={i} 
                          size={12} 
                          className={`${i < Math.floor(garment.rating) ? 'text-orange-400 fill-current' : 'text-gray-300'}`} 
                        />
                      ))}
                    </div>
                    <span className="text-xs text-gray-500 ml-1">({garment.reviews})</span>
                  </div>
                  
                  <div className="flex items-center space-x-2">
                    <span className="text-lg font-semibold text-gray-900">{garment.price}</span>
                    {garment.originalPrice && (
                      <span className="text-sm text-gray-500 line-through">{garment.originalPrice}</span>
                    )}
                  </div>
                  
                  {garment.prime && (
                    <div className="text-xs text-blue-600 mt-1">FREE delivery</div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Product Detail View - PromptDresser özellikleri ile
  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="bg-gray-900 text-white">
        <div className="px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="text-xl font-bold">amazon</div>
              <div className="text-sm">.co.uk</div>
            </div>
            <div className="flex-1 max-w-2xl mx-4">
              <div className="flex">
                <select className="bg-gray-200 text-gray-800 px-3 py-2 rounded-l border-r border-gray-300">
                  <option>All Categories</option>
                </select>
                <input
                  type="text"
                  placeholder="Search Amazon.co.uk"
                  className="flex-1 px-4 py-2 text-gray-800"
                />
                <button className="bg-orange-400 hover:bg-orange-500 px-4 py-2 rounded-r">
                  <Search size={20} />
                </button>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="text-sm">
                <div>Hello, Sign in</div>
                <div className="font-bold">Account & Lists</div>
              </div>
              <div className="text-sm">
                <div>Returns</div>
                <div className="font-bold">& Orders</div>
              </div>
              <div className="flex items-center">
                <ShoppingCart size={24} />
                <span className="font-bold">Basket</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Breadcrumb */}
      <div className="bg-gray-50 px-4 py-2 text-sm">
        <div className="max-w-7xl mx-auto">
          <button 
            onClick={() => setCurrentView('catalog')}
            className="text-blue-600 hover:underline flex items-center"
          >
            <ArrowLeft size={16} className="mr-1" />
            Back to results
          </button>
        </div>
      </div>

      {/* Product Content */}
      <div className="max-w-7xl mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          
          {/* Left Column - Images & Virtual Try-On */}
          <div className="lg:col-span-5">
            {/* Main Product Image */}
            <div className="mb-4">
              <div className="aspect-[4/5] bg-gray-100 rounded-lg overflow-hidden mb-3 max-w-md">
                <img
                  src={
                    typeof selectedImage === 'string' && selectedImage.startsWith('vton-') 
                      ? tryOnResults[selectedImage.replace('vton-', '')]?.image
                      : selectedGarment?.images?.[selectedImage] || selectedGarment?.image
                  }
                  alt={selectedGarment?.name}
                  className="w-full h-full object-cover"
                />
              </div>
              
              {/* Thumbnail Images - Main product + VTON results */}
              <div className="flex flex-wrap gap-2">
                {/* Original product images */}
                {selectedGarment?.images?.map((img, idx) => (
                  <button
                    key={`original-${idx}`}
                    onClick={() => setSelectedImage(idx)}
                    className={`w-16 h-16 border-2 rounded relative ${
                      selectedImage === idx ? 'border-orange-400' : 'border-gray-200'
                    }`}
                  >
                    <img src={img} alt="" className="w-full h-full object-cover rounded" />
                    <div className="absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs px-1 rounded-b">
                      Original
                    </div>
                  </button>
                ))}
                
                {/* VTON Results Thumbnails */}
                {models.map((model) => {
                  const result = tryOnResults[model.id];
                  const isLoading = loading[model.id];
                  
                  return (
                    <button
                      key={`vton-${model.id}`}
                      onClick={() => result && setSelectedImage(`vton-${model.id}`)}
                      className={`w-16 h-16 border-2 rounded relative ${
                        selectedImage === `vton-${model.id}` ? 'border-blue-400' : 'border-gray-200'
                      } ${!result ? 'opacity-50' : ''}`}
                    >
                      {isLoading ? (
                        <div className="w-full h-full bg-gray-100 flex items-center justify-center rounded">
                          <Loader className="animate-spin h-4 w-4 text-blue-600" />
                        </div>
                      ) : result ? (
                        <>
                          <img
                            src={result.image}
                            alt={`${model.name} result`}
                            className="w-full h-full object-cover rounded"
                          />
                          <div className={`absolute bottom-0 left-0 right-0 bg-opacity-80 text-white text-xs px-1 rounded-b ${
                            model.id === 'promptdresser' ? 'bg-red-600' : 'bg-blue-600'
                          }`}>
                            {model.name}
                          </div>
                        </>
                      ) : (
                        <div className="w-full h-full bg-gray-50 flex flex-col items-center justify-center rounded">
                          <div className={`p-1 rounded ${model.color} text-white mb-1`}>
                            {React.cloneElement(model.icon, { size: 12 })}
                          </div>
                          <span className="text-xs">Try On</span>
                        </div>
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Virtual Try-On Controls */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 mb-3 flex items-center">
                <Sparkles size={20} className="mr-2" />
                Virtual Try-On
              </h3>
              
              {!userImage ? (
                <div className="text-center py-4">
                  <Camera size={32} className="text-blue-400 mx-auto mb-2" />
                  <p className="text-sm text-blue-700 mb-3">Upload your photo to try on this item</p>
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleUserImageUpload}
                    accept="image/*"
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded text-sm font-medium"
                  >
                    Upload Photo
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center space-x-2 mb-3">
                    <img src={userImage} alt="Your photo" className="w-8 h-8 rounded-full object-cover" />
                    <span className="text-sm text-green-700 font-medium">Photo uploaded ✓</span>
                  </div>
                  
                  {models.map((model) => {
                    const result = tryOnResults[model.id];
                    const isLoading = loading[model.id];
                    
                    return (
                      <div key={model.id} className="flex items-center justify-between bg-white p-3 rounded border">
                        <div className="flex items-center space-x-3">
                          <div className={`p-2 rounded ${model.color} text-white`}>
                            {model.icon}
                          </div>
                          <div>
                            <div className="font-medium text-sm">{model.name}</div>
                            <div className="text-xs text-gray-500">{model.type}</div>
                            {model.id === 'promptdresser' && result?.usedStyle && (
                              <div className="text-xs text-red-600 mt-1">Style: {result.usedStyle}</div>
                            )}
                          </div>
                        </div>
                        
                        <div className="flex items-center space-x-2">
                          {isLoading ? (
                            <div className="flex items-center text-blue-600">
                              <Loader className="animate-spin h-4 w-4 mr-1" />
                              <span className="text-xs">Processing...</span>
                            </div>
                          ) : result ? (
                            <button
                              onClick={() => tryOnWithModel(selectedGarment, model.id)}
                              className="text-xs text-blue-600 hover:underline"
                            >
                              Retry
                            </button>
                          ) : (
                            <button
                              onClick={() => tryOnWithModel(selectedGarment, model.id)}
                              className="bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded text-xs font-medium"
                            >
                              Try On
                            </button>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          {/* Middle Column - Product Details */}
          <div className="lg:col-span-4">
            <div className="space-y-4">
              {/* Product Title */}
              <div>
                <h1 className="text-2xl font-normal text-gray-900 mb-1">{selectedGarment?.name}</h1>
                <p className="text-blue-600 hover:underline cursor-pointer">Visit the {selectedGarment?.brand} Store</p>
              </div>

              {/* Rating */}
              <div className="flex items-center space-x-2">
                <div className="flex items-center">
                  {[...Array(5)].map((_, i) => (
                    <Star 
                      key={i} 
                      size={16} 
                      className={`${i < Math.floor(selectedGarment?.rating || 0) ? 'text-orange-400 fill-current' : 'text-gray-300'}`} 
                    />
                  ))}
                </div>
                <span className="text-blue-600 hover:underline cursor-pointer text-sm">
                  {selectedGarment?.rating} out of 5 stars
                </span>
                <span className="text-blue-600 hover:underline cursor-pointer text-sm">
                  {selectedGarment?.reviews} ratings
                </span>
              </div>

              <hr className="border-gray-300" />

              {/* Price */}
              <div className="space-y-1">
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600">Price:</span>
                  <span className="text-2xl text-red-600">{selectedGarment?.price}</span>
                  {selectedGarment?.originalPrice && (
                    <span className="text-sm text-gray-500 line-through">{selectedGarment?.originalPrice}</span>
                  )}
                </div>
                {selectedGarment?.prime && (
                  <div className="flex items-center text-sm text-blue-600">
                    <span className="bg-blue-600 text-white px-1 text-xs rounded mr-2">prime</span>
                    FREE delivery
                  </div>
                )}
              </div>

              <hr className="border-gray-300" />

              {/* Size Selection */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">Size:</h3>
                <div className="grid grid-cols-6 gap-2">
                  {selectedGarment?.sizes?.map((size) => (
                    <button
                      key={size}
                      onClick={() => setSelectedSize(size)}
                      className={`py-2 text-sm border rounded ${
                        selectedSize === size
                          ? 'border-orange-400 bg-orange-50'
                          : 'border-gray-300 hover:border-gray-400'
                      }`}
                    >
                      {size}
                    </button>
                  ))}
                </div>
              </div>

              {/* Features */}
              <div>
                <h3 className="font-semibold text-gray-900 mb-2">About this item</h3>
                <ul className="list-disc list-inside text-sm text-gray-700 space-y-1">
                  {selectedGarment?.features?.map((feature, idx) => (
                    <li key={idx}>{feature}</li>
                  ))}
                </ul>
              </div>

              {/* PromptDresser - Detected Style Info */}
              {stylePrompts && stylePrompts.style_options && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <h4 className="font-medium text-red-900 mb-2 flex items-center">
                    <MessageSquare size={16} className="mr-2" />
                    AI Detected Style Info
                  </h4>
                  <div className="text-sm text-red-800 space-y-1">
                    <div><span className="font-medium">Material:</span> {stylePrompts.style_options.material}</div>
                    <div><span className="font-medium">Neckline:</span> {stylePrompts.style_options.neckline}</div>
                    <div><span className="font-medium">Sleeve:</span> {stylePrompts.style_options.sleeve}</div>
                    <div><span className="font-medium">Fit:</span> {stylePrompts.style_options.fit}</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Purchase Options & PromptDresser Style Features */}
          <div className="lg:col-span-3">
            {/* Purchase Box */}
            <div className="border border-gray-300 rounded-lg p-4 space-y-4">
              <div className="text-2xl text-red-600 font-normal">{selectedGarment?.price}</div>
              
              {selectedGarment?.prime && (
                <div className="flex items-center text-sm">
                  <Truck size={16} className="text-blue-600 mr-2" />
                  <span>FREE delivery <strong>Tomorrow</strong></span>
                </div>
              )}
              
              <div className="flex items-center text-sm text-green-700">
                <CheckCircle size={16} className="mr-2" />
                In Stock
              </div>
              
              <div className="space-y-3">
                <div>
                  <label className="text-sm text-gray-700">Quantity:</label>
                  <select 
                    value={quantity}
                    onChange={(e) => setQuantity(e.target.value)}
                    className="ml-2 border border-gray-300 rounded px-2 py-1"
                  >
                    {[1,2,3,4,5].map(num => (
                      <option key={num} value={num}>{num}</option>
                    ))}
                  </select>
                </div>
                
                <button className="w-full bg-yellow-400 hover:bg-yellow-500 text-gray-900 py-2 px-4 rounded font-medium">
                  Add to Basket
                </button>
                
                <button className="w-full bg-orange-400 hover:bg-orange-500 text-white py-2 px-4 rounded font-medium">
                  Buy Now
                </button>
              </div>
              
              <div className="border-t pt-4 space-y-2 text-sm text-gray-600">
                <div className="flex items-center">
                  <Shield size={16} className="mr-2" />
                  Secure transaction
                </div>
                <div className="flex items-center">
                  <RotateCcw size={16} className="mr-2" />
                  Return policy
                </div>
                <div className="flex items-center">
                  <MapPin size={16} className="mr-2" />
                  Dispatch from Amazon
                </div>
              </div>
            </div>

            {/* Upload Photo for Try-On */}
            {!userImage && (
              <div className="mt-6 border border-blue-300 rounded-lg p-4 bg-blue-50">
                <h3 className="font-semibold text-blue-900 mb-2 flex items-center">
                  <Camera size={20} className="mr-2" />
                  Virtual Try-On
                </h3>
                <p className="text-sm text-blue-700 mb-3">
                  See how this item looks on you with AI technology
                </p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleUserImageUpload}
                  accept="image/*"
                  className="hidden"
                />
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded font-medium"
                >
                  Upload Your Photo
                </button>
              </div>
            )}

            {/* PromptDresser Style Selection - 3 Options */}
            <div className="mt-6 border border-red-300 rounded-lg p-4 bg-red-50">
              <h3 className="font-semibold text-red-900 mb-3 flex items-center">
                <Shirt size={20} className="mr-2" />
                PromptDresser Style Options
              </h3>
              
              {stylePrompts && stylePrompts.style_variations ? (
                <div className="grid grid-cols-1 gap-3">
                  {stylePrompts.style_variations.map((variation, idx) => (
                    <button
                      key={idx}
                      onClick={() => setSelectedStyleVariation(idx)}
                      className={`p-3 border rounded-lg text-left transition-all ${
                        selectedStyleVariation === idx
                          ? 'border-red-500 bg-red-100 shadow-md'
                          : 'border-red-200 hover:border-red-300 hover:bg-red-50'
                      }`}
                    >
                      <div className="flex items-start space-x-3">
                        {/* Style Preview Icon */}
                        <div className={`w-12 h-12 rounded border-2 flex items-center justify-center ${
                          selectedStyleVariation === idx ? 'border-red-400 bg-red-200' : 'border-red-300 bg-red-100'
                        }`}>
                          <Shirt size={20} className="text-red-600" />
                        </div>
                        
                        {/* Style Info */}
                        <div className="flex-1">
                          <div className="font-medium text-red-900 text-sm mb-1">
                            {variation.name}
                          </div>
                          <div className="text-xs text-red-700 mb-2">
                            {variation.description}
                          </div>
                          <div className="text-xs text-red-600 space-y-1">
                            <div><span className="font-medium">Tucking:</span> {variation.tucking}</div>
                            <div><span className="font-medium">Fit:</span> {variation.fit}</div>
                            <div><span className="font-medium">Sleeves:</span> {variation.sleeve_rolling.includes('rolled up') ? 'Rolled up' : 'Down'}</div>
                          </div>
                        </div>
                        
                        {/* Selection indicator */}
                        {selectedStyleVariation === idx && (
                          <div className="w-4 h-4 bg-red-500 rounded-full flex items-center justify-center">
                            <div className="w-2 h-2 bg-white rounded-full"></div>
                          </div>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-red-700">
                  <Shirt size={32} className="mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Loading style options...</p>
                </div>
              )}
              
              <div className="mt-4 pt-3 border-t border-red-200">
                <button 
                  onClick={() => userImage && tryOnWithModel(selectedGarment, 'promptdresser')}
                  disabled={!userImage || loading.promptdresser}
                  className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-400 text-white py-2 px-4 rounded text-sm font-medium flex items-center justify-center"
                >
                  {loading.promptdresser ? (
                    <>
                      <Loader className="animate-spin h-4 w-4 mr-2" />
                      Generating Style...
                    </>
                  ) : (
                    <>
                      <Shirt className="h-4 w-4 mr-2" />
                      Try PromptDresser Style
                    </>
                  )}
                </button>
                
                {/* Quick style test buttons */}
                <div className="grid grid-cols-3 gap-2 mt-3">
                  {stylePrompts?.style_variations?.map((variation, idx) => (
                    <button
                      key={idx}
                      onClick={() => {
                        setSelectedStyleVariation(idx);
                        if (userImage) {
                          tryOnWithModel(selectedGarment, 'promptdresser');
                        }
                      }}
                      disabled={!userImage || loading.promptdresser}
                      className="text-xs bg-red-100 hover:bg-red-200 disabled:bg-gray-100 text-red-800 py-1 px-2 rounded border"
                    >
                      {variation.name.split(' ')[0]}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TryOnApp;