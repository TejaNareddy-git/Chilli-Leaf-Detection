// $(document).ready(function () {
//   // -[Scroll Animation]---------------------------
//   $(".navbar a, footer a[href='#halamanku']").on("click", function (event) {
//     if (this.hash !== "") {
//       event.preventDefault();
//       var hash = this.hash;
//       $("html, body").animate(
//         {
//           scrollTop: $(hash).offset().top,
//         },
//         900,
//         function () {
//           window.location.hash = hash;
//         }
//       );
//     }
//   });

//   $(window).scroll(function () {
//     $(".slideanim").each(function () {
//       var pos = $(this).offset().top;
//       var winTop = $(window).scrollTop();
//       if (pos < winTop + 600) {
//         $(this).addClass("slide");
//       }
//     });
//   });

//   // -[model prediction]---------------------------
//   //function to call api
//   $("#prediksi_submit").click(function (e) {
//     e.preventDefault();

//     // Get File img file uploaded by the user
//     var file_data = $("#input_gambar").prop("files")[0];
//     var pics_data = new FormData();
//     pics_data.append("file", file_data);

//     // Call the API with 1 sec timeout (1000 ms)
//     setTimeout(function () {
//       try {
//         $.ajax({
//           url: "/api/deteksi",
//           type: "POST",
//           data: pics_data,
//           processData: false,
//           contentType: false,
//           success: function (res) {
           
//             res_data_prediksi = res["prediksi"];
//             res_gambar_prediksi = res["gambar_prediksi"];
//             res_confidence = res["confidence"];

            
//             generate_prediksi(
//               res_data_prediksi,
//               res_gambar_prediksi,
//               res_confidence
//             );

//             // Process route to get activation visualizations
//             $.ajax({
//               url: "/process",
//               type: "POST",
//               data: pics_data,
//               processData: false,
//               contentType: false,
//               success: function (visualizations) {
//                 display_visualizations(visualizations);
//               },
//               error: function (err) {
//                 console.log("Error fetching activations: ", err);
//               },
//             });
//           },
//         });
//       } catch (e) {
//         // If API fails, log the error in the console
//         console.log("Gagal !");
//         console.log(e);
//       }
//     }, 1000);
//   });

//   // Function tyo display the model prediction result
//   function generate_prediksi(data_prediksi, image_prediksi, confidence) {
//     var str = "";

//     if (image_prediksi == "(none)") {
//       str += "<h3>Prediction Result </h3>";
//       str += "<br>";
//       str += "<h4>Please upload an img file (.jpg)</h4>";
//     } else {
//       str += "<h3>Prediction Result </h3>";
//       str += "<br>";
//       str += "<img src='" + image_prediksi + '\' width="200"></img>';

//       // If confidence is less than 75% display sorry
//       if (confidence < 0.7) {
//         str += "<h4>Sorry, Image cannot be predicted</h4>";
//       } else {
        
//         str += "<h3>" + data_prediksi + "</h3>";
//         str += "<h4>Confidence: " + (confidence * 100).toFixed(2) + "%</h4>";
//       }
//     }
//     $("#hasil_prediksi").html(str);
//   }

//   // Function to display activation visualization
//   function display_visualizations(visualizations) {
//     var visualizationsHTML = "<h3>Model Activation Visualizations</h3>";

//     for (var layerName in visualizations) {
//       var encodedImage = visualizations[layerName];
//       visualizationsHTML += "<h4>" + layerName + "</h4>";
//       visualizationsHTML +=
//         "<img src='data:image/png;base64," +
//         encodedImage +
//         "' width='300' /> <br>";
//     }

//     $("#visualizations").html(visualizationsHTML);
//   }
// });



$(document).ready(function () {
  // -[Scroll Animation]---------------------------
  $(".navbar a, footer a[href='#halamanku']").on("click", function (event) {
    if (this.hash !== "") {
      event.preventDefault();
      var hash = this.hash;
      $("html, body").animate(
        {
          scrollTop: $(hash).offset().top,
        },
        900,
        function () {
          window.location.hash = hash;
        }
      );
    }
  });

  $(window).scroll(function () {
    $(".slideanim").each(function () {
      var pos = $(this).offset().top;
      var winTop = $(window).scrollTop();
      if (pos < winTop + 600) {
        $(this).addClass("slide");
      }
    });
  });

  // -[Model Prediction]---------------------------
  $("#prediksi_submit").click(function (e) {
    e.preventDefault();

    // Get uploaded file
    var file_data = $("#input_gambar").prop("files")[0];
    var pics_data = new FormData();
    pics_data.append("file", file_data);

    // Call the API with 1-second timeout
    setTimeout(function () {
      try {
        $.ajax({
          url: "/api/deteksi",
          type: "POST",
          data: pics_data,
          processData: false,
          contentType: false,
          success: function (res) {
            const res_data_prediksi = res["prediksi"];
            const res_gambar_prediksi = res["gambar_prediksi"];
            const res_confidence = res["confidence"];
            const pesticide_suggestion = res["pesticide_suggestion"];

            // Generate prediction result
            generate_prediksi(
              res_data_prediksi,
              res_gambar_prediksi,
              res_confidence,
              pesticide_suggestion
            );

            // Process route to get activation visualizations
            $.ajax({
              url: "/process",
              type: "POST",
              data: pics_data,
              processData: false,
              contentType: false,
              success: function (visualizations) {
                display_visualizations(visualizations);
              },
              error: function (err) {
                console.error("Error fetching activations: ", err);
              },
            });
          },
          error: function (xhr) {
            alert(`Error: ${xhr.responseJSON.error || "Unknown error occurred"}`);
          },
        });
      } catch (e) {
        console.error("API Call Failed: ", e);
      }
    }, 1000);
  });

  // Function to display the prediction result
  function generate_prediksi(
    data_prediksi,
    image_prediksi,
    confidence,
    pesticide_suggestion
  ) {
    let str = "";

    if (image_prediksi == "(none)") {
      str += "<h3>Prediction Result</h3>";
      str += "<br>";
      str += "<h4>Please upload an image file (.jpg, .png)</h4>";
    } else {
      str += "<h3>Prediction Result</h3>";
      str += "<br>";
      str += "<img src='" + image_prediksi + "' width='200' alt='Prediction Image'/>";

      if (confidence < 0.7) {
        str += "<h4>Sorry, the image could not be confidently predicted.</h4>";
      } else {
        str += `<h3>${data_prediksi}</h3>`;
        str += `<h4>Confidence: ${(confidence * 100).toFixed(2)}%</h4>`;

        if (data_prediksi !== "Healthy") {
          str += `<h4 style="background-color:yellow; margin:8px 0px;"><strong>Pesticide Suggestion:</strong> ${pesticide_suggestion}</h4>`;
        } else {
          str += "<h4>The plant is healthy. No pesticide required.</h4>";
        }
      }
    }
    $("#hasil_prediksi").html(str);
  }

  // Function to display activation visualization
  function display_visualizations(visualizations) {
    let visualizationsHTML = "<h3>Model Activation Visualizations</h3>";

    for (let layerName in visualizations) {
      let encodedImage = visualizations[layerName];
      visualizationsHTML += `<h4>${layerName}</h4>`;
      visualizationsHTML += `<img src='data:image/png;base64,${encodedImage}' width='300' alt='Visualization Image'/><br>`;
    }

    $("#visualizations").html(visualizationsHTML);
  }
});
