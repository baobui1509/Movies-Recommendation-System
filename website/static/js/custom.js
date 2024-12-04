const inputElm = document.querySelector('#genre');
const resultsElm = document.querySelector('#results')

  const swiper = new Swiper('.mySwiper', {
    loop: true,
    spaceBetween: 20,
    autoplay: {
      delay: 3000,
      disableOnInteraction: false,
    },
    pagination: {
      el: '.swiper-pagination',
      clickable: true,
    },
    navigation: {
      nextEl: '.swiper-button-next',
      prevEl: '.swiper-button-prev',
    },
    breakpoints: {
      // Khi màn hình >= 320px
      320: {
        slidesPerView: 1,
        spaceBetween: 10,
      },
      // Khi màn hình >= 768px
      768: {
        slidesPerView: 2,
        spaceBetween: 15,
      },
      // Khi màn hình >= 1024px
      1024: {
        slidesPerView: 4,
        spaceBetween: 20,
      },
    },
    watchOverflow: false,
  });


document.addEventListener('DOMContentLoaded', function () {
  if (inputElm) {
      const tagify = new Tagify(inputElm, {
          enforceWhitelist: true, // Cho phép thêm thể loại ngoài whitelist
          whitelist: [], 
          dropdown: {
              enabled: 1, // Hiển thị dropdown khi người dùng nhập
              maxItems: 10, // Tối đa 10 gợi ý
              closeOnSelect: false,
          },
          maxTags: 5
      });

      // Lấy dữ liệu từ API
      const fetchWhitelist = (query = '') => {
          return fetch(`/genres/whitelist/?query=${query}`)
              .then(response => response.json())
              .then(data => data);
      };

      // Gắn sự kiện khi người dùng nhập
      tagify.on('input', async function (e) {
          const query = e.detail.value.trim();
          console.log('query: ', query);
          tagify.loading(true); // Hiển thị loader

          if (query === '') {
              // Nếu input trống, hiển thị tất cả các thể loại
              const allGenres = await fetchWhitelist();
              console.log('allGenres:', allGenres);
              tagify.whitelist = allGenres;
              tagify.loading(false);
              if (allGenres.length > 0) {
                  tagify.dropdown.show();
              } else {
                  tagify.dropdown.hide();
              }
              
          } else {
              // Nếu có query, hiển thị các thể loại phù hợp
              const filteredGenres = await fetchWhitelist(query);
              console.log('Filtered Genres:', filteredGenres);
              tagify.whitelist = filteredGenres;
              tagify.loading(false);

              if (filteredGenres.length > 0) {
                  tagify.dropdown.show();
              } else {
                  tagify.dropdown.hide();
              }
          }

          // tagify.loading(false);
      });

      tagify.on('add', function () {
          fetchWhitelist().then(allGenres => {
            console.log('add: ', allGenres)
            tagify.whitelist = allGenres;
            tagify.loading(false);
            tagify.dropdown.show();
        });
    });

      tagify.on('focus', async function () {
          // Khi click vào ô input, hiển thị tất cả các thể loại
          const allGenres = await fetchWhitelist();
          tagify.whitelist = allGenres;
          if (allGenres.length > 0) {
              tagify.dropdown.show();
          }
      });

      // Nút xóa tất cả thẻ
      document.querySelector('.tags--removeAllBtn')
          .addEventListener('click', tagify.removeAllTags.bind(tagify));
  }
});

if (resultsElm){
  console.log('resultsElm: ',resultsElm.childElementCount)
  if (resultsElm.childElementCount !== 0) {
    setTimeout(function() {
      resultsElm.scrollIntoView({
          behavior: 'smooth' 
      });
  }, 300); 
  }
    
}

// function scrollToResults() {
//   window.location.hash = "#results"; 
// }





