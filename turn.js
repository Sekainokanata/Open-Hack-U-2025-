const image = document.querySelector('.unko');


image.addEventListener('keydown', event =>{
    if(event.code == "KeyR"){
        image.animate(
            [{transform: 'rotate(0deg)'}, {transform:'rotate(360deg)'}],
            {fill:'backwards', duration: 1000,},
        );
        
    }else if(event.code == "KeyL"){
        image.animate(
            [{transform: 'rotate(360deg)'}, {transform:'rotate(0deg)'}],
            {fill:'backwards', duration: 1000,},
        );
    }
});