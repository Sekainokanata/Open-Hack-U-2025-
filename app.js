const image = document.querySelector('.unko');

// 画像にマウスが乗った時に振動開始
image.addEventListener('mouseenter', () => {
    image.classList.add('shaking');
});

// 画像からマウスが外れた時に振動停止
image.addEventListener('mouseleave', () => {
    image.classList.remove('shaking');
});

// 画像をクリックしたときにも振動
image.addEventListener('click', () => {
    image.classList.add('shaking');
    // クリック後に1秒間だけ振動させて停止
    setTimeout(() => {
    image.classList.remove('shaking');
    }, 1000);
});