$(document).ready(function(){


$(function () {
  count = 0;
  wordsArray = ["jawdropping", "illegal", "bad"];
  setInterval(function () {
    count++;
    $("#word").fadeOut(300, function () {
      $(this).text(wordsArray[count % wordsArray.length]).fadeIn(300);
    });
  }, 2000);
});

});
// // Variables
