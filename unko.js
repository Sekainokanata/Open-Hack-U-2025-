// 画像を表示するためのコンテナを取得
const imageContainer = document.getElementById('imageContainer');
let imageCount = 0; // 画像の数を管理するカウンター

// 画像を追加する関数
function addImage() {
  // 新しいimg要素を作成
  const newImage = document.createElement('img');
  newImage.src = 'https://thumb.ac-illust.com/5d/5dd51d6f6e1bddd606b367e2de8cba1a_w.jpeg'; // ここに画像のURLを指定
  newImage.alt = `画像 ${imageCount + 1}`;
  newImage.id = `image-${imageCount}`; // 画像に一意のIDを設定
  imageContainer.appendChild(newImage);
  imageCount++; // カウンターを増加
}

// 画像を削除する関数
function removeImage() {
  if (imageCount > 0) {
    const lastImage = document.getElementById(`image-${imageCount - 1}`);
    imageContainer.removeChild(lastImage); // 最後に追加された画像を削除
    imageCount--; // カウンターを減少
  }
}

// ボタンのクリックイベントに関数を紐付け
document.getElementById('addImageButton').addEventListener('click', addImage);
document.getElementById('removeImageButton').addEventListener('click', removeImage);