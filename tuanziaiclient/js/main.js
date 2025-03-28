// 初始化 Mapbox
mapboxgl.accessToken = 'pk.eyJ1IjoiYXRmaWVsZDIwMjIiLCJhIjoiY2xlZjFodW1lMDR3dTNvbXVvajMwNGxzZSJ9.1FnjGYOuY7l-Us1SFatgKg';
const map = new mapboxgl.Map({
    container: 'map', // 地图块的 id
    style: 'mapbox://styles/mapbox/streets-v11', // 地图样式
    center: [121.4737, 31.2304], // 初始中心坐标 (上海)
    zoom: 10 // 缩放级别
});
