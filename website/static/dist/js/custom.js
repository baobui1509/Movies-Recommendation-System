// const swiper = new Swiper('.mySwiper', {
//     loop: true, 
//     autoplay: {
//       delay: 3000,
//       disableOnInteraction: false, 
//     },
//     pagination: {
//       el: '.swiper-pagination',
//       clickable: true,
//     },
//     navigation: {
//       nextEl: '.swiper-button-next',
//       prevEl: '.swiper-button-prev',
//     },
//   });

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
  
  
  