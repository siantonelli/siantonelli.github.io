// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "About",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-publications",
          title: "Publications",
          description: "Thumbnail previews are hallucinated by AI (ChatGPT &amp;#8482;). * denotes equal contribution.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/publications/";
          },
        },{id: "nav-repositories",
          title: "Repositories",
          description: "GitHub statistics",
          section: "Navigation",
          handler: () => {
            window.location.href = "/repositories/";
          },
        },{id: "nav-cv",
          title: "CV",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "post-google-gemini-updates-flash-1-5-gemma-2-and-project-astra",
        
          title: 'Google Gemini updates: Flash 1.5, Gemma 2 and Project Astra <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "We’re sharing updates across our Gemini family of models and a glimpse of Project Astra, our vision for the future of AI assistants.",
        section: "Posts",
        handler: () => {
          
            window.open("https://blog.google/technology/ai/google-gemini-update-flash-ai-assistant-io-2024/", "_blank");
          
        },
      },{id: "post-displaying-external-posts-on-your-al-folio-blog",
        
          title: 'Displaying External Posts on Your al-folio Blog <svg width="1.2rem" height="1.2rem" top=".5rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>',
        
        description: "",
        section: "Posts",
        handler: () => {
          
            window.open("https://medium.com/@al-folio/displaying-external-posts-on-your-al-folio-blog-b60a1d241a0a?source=rss-17feae71c3c4------2", "_blank");
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/books/the_godfather/";
            },},{id: "news-our-survey-has-been-accepted-at-acm-computing-surveys",
          title: 'Our survey has been accepted at ACM Computing Surveys.',
          description: "",
          section: "News",},{id: "news-moving-to-saarbrücken-to-join-aleksandar-bojchevski-s-group-as-a-ph-d-student",
          title: 'Moving to Saarbrücken to join Aleksandar Bojchevski’s group as a Ph.D. student 🇩🇪...',
          description: "",
          section: "News",},{id: "news-our-paper-on-few-shot-graph-classification-has-been-accepted-at-learning-on-graphs-log-conference",
          title: 'Our paper on few-shot graph classification has been accepted at Learning on Graphs...',
          description: "",
          section: "News",},{id: "news-i-have-been-selected-as-an-outstanding-graduate-student-for-the-a-y-2020-2021",
          title: 'I have been selected as an outstanding graduate student for the a.y. 2020/2021...',
          description: "",
          section: "News",},{id: "news-our-paper-on-conformal-prediction-for-graphs-has-been-accepted-at-icml",
          title: 'Our paper on conformal prediction for graphs has been accepted at ICML 🌺...',
          description: "",
          section: "News",},{id: "news-moving-to-cologne",
          title: 'Moving to Cologne 🎡',
          description: "",
          section: "News",},{id: "news-admitted-to-attend-london-geometry-and-machine-learning-logml-summer-school-working-on-gnns-for-relational-databases",
          title: 'Admitted to attend London Geometry and Machine Learning (LOGML) Summer School working on...',
          description: "",
          section: "News",},{id: "news-admitted-to-attend-ellis-doctoral-symposium-eds",
          title: 'Admitted to attend ELLIS Doctoral Symposium (EDS) 🇫🇷',
          description: "",
          section: "News",},{id: "news-i-will-join-amboss-tech-for-a-3-months-internship",
          title: 'I will join Amboss Tech for a 3-months internship 👨‍💻🇺🇸',
          description: "",
          section: "News",},{id: "news-i-have-been-awarded-the-best-poster-at-eds-for-my-work-on-data-valuation-for-graphs",
          title: 'I have been awarded the best poster at EDS for my work on...',
          description: "",
          section: "News",},{id: "news-my-work-on-data-valuation-for-graphs-has-been-accepted-at-the-2nd-workshop-on-attributing-model-behavior-at-scale-attrib-neurips",
          title: 'My work on Data Valuation for Graphs has been accepted at the 2nd...',
          description: "",
          section: "News",},{id: "news-i-will-be-attending-the-log-meetup-in-aachen",
          title: 'I will be attending the LOG meetup in Aachen 🇩🇪',
          description: "",
          section: "News",},{id: "news-i-will-be-attending-the-mediterranean-machine-learning-m2l-summer-school",
          title: 'I will be attending the Mediterranean Machine Learning (M2L) Summer School 🇭🇷',
          description: "",
          section: "News",},{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/siantonelli", "_blank");
        },
      },{
        id: 'social-x',
        title: 'X',
        section: 'Socials',
        handler: () => {
          window.open("https://twitter.com/sntonelli", "_blank");
        },
      },{
        id: 'social-bluesky',
        title: 'Bluesky',
        section: 'Socials',
        handler: () => {
          window.open("https://bsky.app/profile/siantonelli.bsky.social", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/siantonelli", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=V1ECJ8YAAAAJ", "_blank");
        },
      },];
