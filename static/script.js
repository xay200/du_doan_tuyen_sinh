// Hàm hiển thị Loading khi submit
function showLoader() {
    const loader = document.getElementById('loader');
    loader.style.display = 'flex';
}

function updateSubjects() {
    const block = document.getElementById("blockSelect").value;
    
    // Mảng tất cả các môn tự chọn
    const subjects = ["physics", "chemistry", "biology", "history", "geography", "gdcd"];
    
    // 1. Reset: Ẩn tất cả và bỏ required
    subjects.forEach(sub => {
        const wrapper = document.getElementById("wrap_" + sub);
        const input = document.getElementById(sub);
        
        if(wrapper) wrapper.classList.add("hidden");
        if(input) input.required = false; 
    });

    // 2. Hàm helper để hiện môn
    const show = (ids) => {
        ids.forEach(id => {
            const wrapper = document.getElementById("wrap_" + id);
            const input = document.getElementById(id);
            
            if(wrapper) {
                wrapper.classList.remove("hidden");
                wrapper.classList.add("fade-in"); // Thêm hiệu ứng hiện dần
            }
            if(input) input.required = true;
        });
    }

    // 3. Logic hiển thị theo khối
    if (block === "A00") show(["physics", "chemistry"]);
    else if (block === "A01") show(["physics"]);
    else if (block === "B00") show(["chemistry", "biology"]);
    else if (block === "C00") show(["history", "geography"]);
    // D01 thì chỉ Toán Văn Anh (đã hiện sẵn)
}