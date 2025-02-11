// 基本画像設定
const imageSrc = "https://thumb.ac-illust.com/5d/5dd51d6f6e1bddd606b367e2de8cba1a_w.jpeg"; // 元画像のソースを取得

// HTMLで表示する要素を取得
const container = document.createElement("div");
container.style.position = "relative";
container.style.width = "100%";
container.style.height = "100vh";
container.style.overflow = "hidden";
document.body.appendChild(container);

// 左クリックで画像追加
container.addEventListener("click", (event) => {
  const image = document.createElement("img");
  image.src = imageSrc;
  image.classList.add("unko");
  image.style.width = "50px";
  image.style.height = "50px";
  image.style.position = "absolute";
  image.style.left = `${event.clientX - 25}px`;
  image.style.top = `${event.clientY - 25}px`;
  image.style.cursor = "move";
  container.appendChild(image);
});

// 右クリックで画像削除
container.addEventListener("contextmenu", (event) => {
  event.preventDefault(); // デフォルトの右クリックメニューを無効化
  const elements = document.elementsFromPoint(event.clientX, event.clientY);
  for (const element of elements) {
    if (element.tagName === "IMG" && element.classList.contains("unko")) {
      element.remove();
      break;
    }
  }
});

